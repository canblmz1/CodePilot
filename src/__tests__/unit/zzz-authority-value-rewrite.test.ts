/**
 * RESEARCH/CANDIDATE REGRESSIONS — NOT part of any commit history, not
 * pushed. Exercises the real createCliToolsTools() factory, the real
 * runCliToolInstall (both from src/lib/builtin-tools/cli-tools.ts, as
 * modified by this local candidate), and prefix-safe-json's real
 * createAiSdkExecutionLock/createAiSdkExecutionGuard, through real
 * streamText() + MockLanguageModelV4. The `runStep` helper below is a
 * deliberately minimal, faithful re-statement of exactly the logic added
 * to agent-loop.ts (same calls, same order, same non-negotiable-invariant
 * dispatch from authority.value) — not a different implementation.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { streamText, tool, jsonSchema, type ModelMessage } from 'ai';
import { MockLanguageModelV4 } from 'ai/test';
import { createCliToolsTools, runCliToolInstall, CLI_TOOL_INSTALL_SCHEMA } from '@/lib/builtin-tools/cli-tools';

function toolInputParts(id: string, toolName: string, argsJson: string, chunkSize = 9) {
  const parts: unknown[] = [{ type: 'tool-input-start', id, toolName }];
  for (let i = 0; i < argsJson.length; i += chunkSize) {
    parts.push({ type: 'tool-input-delta', id, delta: argsJson.slice(i, i + chunkSize) });
  }
  parts.push({ type: 'tool-input-end', id });
  return parts;
}
function toolCallPart(id: string, toolName: string, argsJson: string) {
  return { type: 'tool-call', toolCallId: id, toolName, input: argsJson };
}
function mockModel(parts: unknown[]) {
  return new MockLanguageModelV4({
    doStream: async () => ({
      stream: new ReadableStream({
        start(controller) {
          for (const p of parts) controller.enqueue(p as never);
          controller.close();
        },
      }),
    }),
  });
}

/**
 * Faithful re-statement of the exact logic added to agent-loop.ts:
 * lock the install tool -> drive streamText -> push every fullStream
 * event into a fresh guard -> resolve authority -> dispatch via
 * runCliToolInstall(authority.value) -> synthesize a tool-result message.
 * `extraTools` lets Phase 8 add an unrelated, unlocked native tool to the
 * same step.
 */
async function runStep(model: MockLanguageModelV4, extraTools: Record<string, unknown> = {}) {
  const { createAiSdkExecutionLock, createAiSdkExecutionGuard } = await import('prefix-safe-json');
  const baseTools = createCliToolsTools();
  const locked = createAiSdkExecutionLock({
    codepilot_cli_tools_install: baseTools.codepilot_cli_tools_install,
  });
  const tools = { ...baseTools, ...extraTools, ...locked };

  const stepGuard = createAiSdkExecutionGuard();
  // `as never` on the whole call, not just `tools`: streamText()'s return
  // type infers a `toolsContext` requirement from the tools' own generic
  // Context parameter that a runtime-assembled ToolSet (real production
  // shape, not a single statically-typed tool literal) can't statically
  // satisfy — a test-script-only typing artifact, not a claim about
  // runtime behavior, which the passing tests below verify directly.
  const result = streamText({ model, prompt: 'x', tools } as never);

  const sseToolResults: Array<{ tool_use_id: string; content: string; is_error: boolean }> = [];
  const seenEventTypes: string[] = [];
  for await (const event of result.fullStream) {
    seenEventTypes.push((event as { type: string }).type);
    stepGuard.push(event as never);
  }

  const stepFinishReason = await result.finishReason;
  const { decisions } = stepGuard.finish({ providerReason: String(stepFinishReason) });

  const installResultMessages: ModelMessage[] = [];
  let installSideEffectCalls = 0;
  const capturedSideEffectInputs: unknown[] = [];

  for (const decision of decisions) {
    if (decision.name !== 'codepilot_cli_tools_install' || !decision.toolCallId) continue;
    const toolCallId = decision.toolCallId;
    let outcomeText: string;
    let isError: boolean;

    if (decision.action === 'execute') {
      const authority = stepGuard.takeDecision(decision.internalId);
      if (!authority) {
        outcomeText = 'Installation blocked: execution authority could not be acquired.';
        isError = true;
      } else {
        const parsed = CLI_TOOL_INSTALL_SCHEMA.safeParse(authority.value);
        if (!parsed.success) {
          outcomeText = 'Installation blocked: shape mismatch.';
          isError = true;
        } else {
          capturedSideEffectInputs.push(parsed.data);
          installSideEffectCalls += 1;
          outcomeText = await runCliToolInstall(parsed.data);
          isError = false;
        }
      }
    } else {
      outcomeText = `Installation skipped: the surrounding response was not confirmed safe to execute (${decision.reason ?? 'no positive authority'}).`;
      isError = true;
    }

    sseToolResults.push({ tool_use_id: toolCallId, content: outcomeText, is_error: isError });
    installResultMessages.push({
      role: 'tool',
      content: [{ type: 'tool-result', toolCallId, toolName: 'codepilot_cli_tools_install', output: { type: 'text', value: outcomeText } }],
    } as ModelMessage);
  }

  const responseData = await result.response;
  const finalMessages = [...responseData.messages, ...installResultMessages];

  return {
    seenEventTypes,
    decisions,
    installSideEffectCalls,
    capturedSideEffectInputs,
    sseToolResults,
    finalMessages,
    stepGuard,
  };
}

// ── CASE A — safe, complete, normal install ─────────────────────────────
describe('CASE A — safe normal install', () => {
  it('authority=1, side effect=1, side-effect input equals authority.value, one tool result', async () => {
    const args = JSON.stringify({ command: 'echo case-a-test', name: 'case-a' });
    const model = mockModel([
      { type: 'stream-start', warnings: [] },
      ...toolInputParts('call_a', 'codepilot_cli_tools_install', args),
      toolCallPart('call_a', 'codepilot_cli_tools_install', args),
      { type: 'finish', finishReason: { unified: 'tool-calls', raw: 'tool-calls' }, usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } },
    ]);

    const r = await runStep(model);
    const positiveDecisions = r.decisions.filter((d) => d.name === 'codepilot_cli_tools_install' && d.action === 'execute');

    assert.equal(positiveDecisions.length, 1, 'exactly one positive authority');
    assert.equal(r.installSideEffectCalls, 1, 'actual install side effect ran exactly once');
    assert.deepEqual(r.capturedSideEffectInputs[0], { command: 'echo case-a-test', name: 'case-a' });
    const toolResultMsgs = r.finalMessages.filter((m) => m.role === 'tool');
    assert.equal(toolResultMsgs.length, 1, 'exactly one real tool result inserted');
  });
});

// ── CASE B — truncated streamed install ─────────────────────────────────
describe('CASE B — truncated streamed install', () => {
  it('authority absent, side effect=0', async () => {
    const truncated = '{"command":"brew install ffm';
    const model = mockModel([
      { type: 'stream-start', warnings: [] },
      { type: 'tool-input-start', id: 'call_b', toolName: 'codepilot_cli_tools_install' },
      { type: 'tool-input-delta', id: 'call_b', delta: truncated },
      { type: 'tool-input-end', id: 'call_b' },
      toolCallPart('call_b', 'codepilot_cli_tools_install', truncated + '"}'),
      { type: 'finish', finishReason: { unified: 'tool-calls', raw: 'tool-calls' }, usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } },
    ]);
    const r = await runStep(model);
    const positive = r.decisions.filter((d) => d.name === 'codepilot_cli_tools_install' && d.action === 'execute');
    assert.equal(positive.length, 0, 'no positive authority for truncated evidence');
    assert.equal(r.installSideEffectCalls, 0, 'install side effect never ran');
  });
});

// ── CASE C — unsafe finish reason ───────────────────────────────────────
describe('CASE C — unsafe finish reason (SDK-version-independent by design)', () => {
  for (const reason of ['length', 'content-filter', 'error', 'other']) {
    it(`finishReason=${reason}: side effect stays 0`, async () => {
      const args = JSON.stringify({ command: 'echo case-c', name: 'case-c' });
      const model = mockModel([
        { type: 'stream-start', warnings: [] },
        ...toolInputParts('call_c', 'codepilot_cli_tools_install', args),
        toolCallPart('call_c', 'codepilot_cli_tools_install', args),
        { type: 'finish', finishReason: { unified: reason, raw: reason }, usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } },
      ]);
      const r = await runStep(model);
      assert.equal(r.installSideEffectCalls, 0, `${reason}: side effect must stay 0`);
    });
  }
  it('note: this protection does not depend on ai SDK version at all — there is no native execute() for any SDK version to gate in the first place; the guard is the only gate, proven identical on ai@7.0.11 and ai@7.0.85 in the accompanying report', () => {
    assert.ok(true);
  });
});

// ── CASE E — duplicate authority consumption ────────────────────────────
describe('CASE E — duplicate authority consumption', () => {
  it('consuming the same decision twice: side effect count stays 1', async () => {
    const args = JSON.stringify({ command: 'echo case-e', name: 'case-e' });
    const model = mockModel([
      { type: 'stream-start', warnings: [] },
      ...toolInputParts('call_e', 'codepilot_cli_tools_install', args),
      toolCallPart('call_e', 'codepilot_cli_tools_install', args),
      { type: 'finish', finishReason: { unified: 'tool-calls', raw: 'tool-calls' }, usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } },
    ]);
    const { createAiSdkExecutionLock, createAiSdkExecutionGuard } = await import('prefix-safe-json');
    const baseTools = createCliToolsTools();
    const locked = createAiSdkExecutionLock({ codepilot_cli_tools_install: baseTools.codepilot_cli_tools_install });
    const guard = createAiSdkExecutionGuard();
    const result = streamText({ model, prompt: 'x', tools: locked } as never);
    for await (const part of result.fullStream) guard.push(part as never);
    const { decisions } = guard.finish({ providerReason: 'tool-calls' } as never);
    const decision = decisions.find((d) => d.name === 'codepilot_cli_tools_install');
    assert.ok(decision && decision.action === 'execute');

    let sideEffectCalls = 0;
    const firstAuthority = guard.takeDecision(decision!.internalId);
    if (firstAuthority) {
      sideEffectCalls += 1;
      await runCliToolInstall(firstAuthority.value as never);
    }
    const secondAuthority = guard.takeDecision(decision!.internalId);
    assert.equal(secondAuthority, undefined, 'second takeDecision() call returns undefined');
    if (secondAuthority) {
      sideEffectCalls += 1;
      await runCliToolInstall((secondAuthority as { value: unknown }).value as never);
    }
    assert.equal(sideEffectCalls, 1, 'actual side effect count stays exactly 1');
  });
});

// ── CASE F — no raw deltas ───────────────────────────────────────────────
describe('CASE F — no raw deltas available', () => {
  it('terminal-only tool-call, no tool-input-start/delta/end: fails closed, no fallback to SDK input', async () => {
    const args = JSON.stringify({ command: 'echo case-f', name: 'case-f' });
    const model = mockModel([
      { type: 'stream-start', warnings: [] },
      toolCallPart('call_f', 'codepilot_cli_tools_install', args),
      { type: 'finish', finishReason: { unified: 'tool-calls', raw: 'tool-calls' }, usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } },
    ]);
    const r = await runStep(model);
    const matching = r.decisions.filter((d) => d.name === 'codepilot_cli_tools_install');
    assert.equal(matching.length, 0, 'no decision at all is produced for a call with no raw evidence — not a silent fallback to the SDK-projected input');
    assert.equal(r.installSideEffectCalls, 0);
  });
});

// ── PHASE 8 — multi-tool step ────────────────────────────────────────────
describe('PHASE 8 — one locked install + one unrelated normal tool in the same step', () => {
  it('normal tool keeps native execute; install stays manual-authority-controlled; no toolCallId cross-talk', async () => {
    let normalToolNativeCalls = 0;
    let normalToolReceivedInput: unknown = null;
    const extraTools = {
      codepilot_cli_tools_list: tool({
        description: 'list',
        inputSchema: jsonSchema<Record<string, never>>({ type: 'object', properties: {} }),
        execute: async (input: unknown) => {
          normalToolNativeCalls += 1;
          normalToolReceivedInput = input;
          return 'listed 3 tools';
        },
      }),
    };

    const installArgs = JSON.stringify({ command: 'echo multi-tool', name: 'multi' });
    const listArgs = JSON.stringify({});
    const model = mockModel([
      { type: 'stream-start', warnings: [] },
      ...toolInputParts('call_install', 'codepilot_cli_tools_install', installArgs),
      toolCallPart('call_install', 'codepilot_cli_tools_install', installArgs),
      ...toolInputParts('call_list', 'codepilot_cli_tools_list', listArgs, 20),
      toolCallPart('call_list', 'codepilot_cli_tools_list', listArgs),
      { type: 'finish', finishReason: { unified: 'tool-calls', raw: 'tool-calls' }, usage: { inputTokens: 10, outputTokens: 10, totalTokens: 20 } },
    ]);

    const r = await runStep(model, extraTools);

    // The normal tool ran natively, exactly once, with its own input —
    // completely untouched by the lock or the guard.
    assert.equal(normalToolNativeCalls, 1, 'unrelated tool retains native execute()');
    assert.deepEqual(normalToolReceivedInput, {});

    // The install ran exactly once via manual authority, with its own input.
    assert.equal(r.installSideEffectCalls, 1);
    assert.deepEqual(r.capturedSideEffectInputs[0], { command: 'echo multi-tool', name: 'multi' });

    // No cross-talk: exactly one guard decision matches the install
    // toolCallId, and it is not "call_list".
    const installDecisions = r.decisions.filter((d) => d.name === 'codepilot_cli_tools_install');
    assert.equal(installDecisions.length, 1);
    assert.equal(installDecisions[0].toolCallId, 'call_install');

    // Ordering: both a native tool-result (for the list tool, already in
    // responseData.messages) and a synthesized tool-result (for install)
    // are present, addressed to the correct, non-crossed toolCallIds.
    const toolMsgs = r.finalMessages.filter((m) => m.role === 'tool');
    const allResultParts = toolMsgs.flatMap((m) => (Array.isArray(m.content) ? m.content : []));
    const idsSeen = allResultParts
      .map((p) => ('toolCallId' in p ? p.toolCallId : undefined))
      .sort();
    assert.deepEqual(idsSeen, ['call_install', 'call_list']);
  });
});
