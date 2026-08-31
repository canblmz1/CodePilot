/**
 * RESEARCH/CANDIDATE REGRESSION — CASE D — NOT part of any commit, not
 * pushed. Real, unmodified @ai-sdk/openai@4.0.52 (installed --no-save,
 * testing the general provider-adapter architecture — not a claim about
 * CodePilot's own currently-pinned @ai-sdk/openai@4.0.5's exact internal
 * layout, which is organized differently release-to-release but shares
 * the same Responses API event contract). A custom `fetch` returns real,
 * schema-valid Responses API SSE bytes; nothing below the HTTP layer is
 * mocked. Drives the REAL createCliToolsTools() install tool through the
 * exact dispatch logic added to agent-loop.ts.
 */
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { streamText } from 'ai';
import { createOpenAI } from '@ai-sdk/openai';
import { createCliToolsTools, runCliToolInstall, CLI_TOOL_INSTALL_SCHEMA } from '@/lib/builtin-tools/cli-tools';

function sse(events: unknown[]) {
  const body = events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join('') + 'data: [DONE]\n\n';
  return new Response(body, { status: 200, headers: { 'content-type': 'text/event-stream' } });
}
function mockFetch(events: unknown[]) {
  return async () => sse(events);
}

let seq = 0;
function n(event: Record<string, unknown>) {
  return { sequence_number: seq++, ...event };
}
const USAGE = { input_tokens: 10, output_tokens: 10 };
const responseId = 'resp_case_d';
const itemId = 'fc_case_d';
const callId = 'call_case_d';

describe('CASE D — OpenAI real-adapter projected-input divergence', () => {
  it('streamed delta evidence truncated (A), final SDK-projected input different (B): native execute=0, side effect never runs on B, authority.value is never even offered because raw evidence never completed', async () => {
    seq = 0;
    const truncatedDelta = '{"command":"brew install ffm';
    // The provider's own separately-reported "done" value — syntactically
    // valid, schema-valid, and NOT what was actually streamed. This is the
    // real @ai-sdk/openai architecture: response.output_item.done's
    // item.arguments is an independently-sourced field, never reconciled
    // against the function_call_arguments.delta stream by the adapter.
    const divergentDoneArgs = '{"command":"brew install ffm"}';

    const events = [
      n({ type: 'response.created', response: { id: responseId, created_at: 0, model: 'gpt-4o' } }),
      n({ type: 'response.output_item.added', output_index: 0, item: { id: itemId, type: 'function_call', call_id: callId, name: 'codepilot_cli_tools_install', arguments: '' } }),
      n({ type: 'response.function_call_arguments.delta', item_id: itemId, output_index: 0, delta: truncatedDelta }),
      n({ type: 'response.function_call_arguments.done', item_id: itemId, output_index: 0, arguments: divergentDoneArgs }),
      n({ type: 'response.output_item.done', output_index: 0, item: { id: itemId, type: 'function_call', status: 'completed', call_id: callId, name: 'codepilot_cli_tools_install', arguments: divergentDoneArgs } }),
      n({ type: 'response.completed', response: { usage: USAGE } }),
    ];

    const openai = createOpenAI({ apiKey: 'test-key', fetch: mockFetch(events) });
    const { createAiSdkExecutionLock, createAiSdkExecutionGuard } = await import('prefix-safe-json');
    const baseTools = createCliToolsTools();
    const locked = createAiSdkExecutionLock({ codepilot_cli_tools_install: baseTools.codepilot_cli_tools_install });

    const nativeExecuteCalls = 0; // control: no execute exists on the locked tool at all — always 0, structurally
    const guard = createAiSdkExecutionGuard();

    const result = streamText({
      model: openai('gpt-4o'),
      prompt: 'irrelevant — mock transport ignores it',
      tools: locked,
    } as never);

    let sawToolCall = false;
    let finalToolCallInput: unknown;
    for await (const part of result.fullStream) {
      guard.push(part as never);
      if ((part as { type: string }).type === 'tool-call') {
        sawToolCall = true;
        finalToolCallInput = (part as { input: unknown }).input;
      }
    }

    const stepFinishReason = await result.finishReason;
    const { decisions } = guard.finish({ providerReason: String(stepFinishReason) });
    const decision = decisions.find((d) => d.name === 'codepilot_cli_tools_install');

    // The SDK genuinely did project a complete-looking (wrong) final input —
    // confirming this is a real reproduction, not a scenario that can't
    // occur. ai core successfully JSON-parses a syntactically valid string
    // at this layer (confirmed elsewhere this session: parseToolCall's
    // safeParseJSON succeeds and the fullStream tool-call part carries the
    // parsed object, not the original string) — B parsed cleanly BECAUSE
    // it is syntactically valid, which is exactly what makes it dangerous:
    // structurally indistinguishable from a genuine, complete call.
    assert.equal(sawToolCall, true);
    assert.deepEqual(finalToolCallInput, JSON.parse(divergentDoneArgs), 'SDK really did project B (parsed successfully), independent of the truncated raw stream A');

    // Native execute=0: trivially true AND structurally guaranteed — the
    // locked tool has no execute field for the SDK to have called at all.
    assert.equal((locked.codepilot_cli_tools_install as { execute?: unknown }).execute, undefined);
    assert.equal(nativeExecuteCalls, 0);

    // The actual invariant under test: does authority ever get granted for
    // this call, and if somehow it did, would the value be B?
    let sideEffectCalls = 0;
    let dispatchedValue: unknown = null;
    if (decision?.action === 'execute') {
      const authority = guard.takeDecision(decision.internalId);
      if (authority) {
        const parsed = CLI_TOOL_INSTALL_SCHEMA.safeParse(authority.value);
        if (parsed.success) {
          dispatchedValue = parsed.data;
          sideEffectCalls += 1;
          await runCliToolInstall(parsed.data);
        }
      }
    }

    // Not necessarily 'reject' specifically — the raw evidence here looks
    // like an in-progress truncation, not confirmed-malformed JSON, so the
    // guard's own decision vocabulary correctly classifies it 'retry'
    // ("nothing wrong with what arrived, no trustworthy complete value
    // yet") rather than 'reject' ("the data itself is the problem"). The
    // invariant under test is narrower and unconditional either way: never
    // 'execute'.
    assert.notEqual(decision?.action, 'execute', 'never authorized to execute merely because the SDK separately projected a complete-looking value');
    assert.equal(sideEffectCalls, 0, 'B is never dispatched to the real side effect merely because the SDK projected it');
    assert.equal(dispatchedValue, null);
  });
});
