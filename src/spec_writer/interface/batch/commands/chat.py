"""``spec-writer chat`` — primary interactive entry-point.

Free-form input goes to the session-level :class:`ChatAgent` (one
agent, one context, one history — see the ownership principle). The agent
decides what to do: read sources, list the corpus, ask for
clarifications, write specs, validate, patch errors, report.

Slash commands stay as **escape-hatches** for direct batch operations
(generate, refine, validate, metrics, status, add-feature,
cascade-fix, edit, refine-plan, reload, session, help, quit). They
bypass the chat agent and call ``run_pipeline_for`` directly.

Streaming: text deltas + tool-call markers are printed to stdout as
they arrive from the LLM. Every event is also persisted to the
``ChatSession`` JSON log under ``_meta/sessions/<id>.json``.

Resume: ``spec-writer chat --resume <session-id>`` reloads a prior
session's history into the agent context so the conversation
continues across runs.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer

from spec_writer.agent.chat_agent import ChatAgent
from spec_writer.agent.chat_agent_factory import (
    UnsupportedProviderError,
    build_chat_agent,
)
from spec_writer.agent.chat_session import ChatSession, load_session
from spec_writer.agent.events import StatusEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from spec_writer.agent.tools.ask_user import set_answerer
from spec_writer.engine.cascade import cascade_neighbours
from spec_writer.engine.project_context import NoProjectRootError, ProjectContext
from spec_writer.engine.refine_plan import (
    build_refine_plan,
    scope_spec_ids,
    select_plan_entries,
)
from spec_writer.interface.batch.commands.generate import (
    _load_existing_corpus,
    _load_methodology_for,
    _report_result,
    run_pipeline_for,
)
from spec_writer.methodology.hot_reload import diff_methodologies
from spec_writer.methodology.loader import MethodologyLoadError
from spec_writer.methodology.registry import ResolvedMethodology
from spec_writer.types import Scope

SlashHandler = Callable[["ChatContext", list[str]], int]


class ChatContext:
    """Mutable state shared by every slash handler and the ChatAgent."""

    def __init__(
        self,
        project_ctx: ProjectContext,
        session: ChatSession,
        default_layers: list[str],
        max_iterations: int,
        methodology: ResolvedMethodology | None = None,
        chat_agent: ChatAgent | None = None,
    ) -> None:
        self.project_ctx = project_ctx
        self.session = session
        self.default_layers = default_layers
        self.max_iterations = max_iterations
        self.methodology = methodology
        self.chat_agent = chat_agent


def run(
    ctx: typer.Context,
    layer: list[str] = typer.Option(
        [], "--layer", help="Default layer(s) for slash commands (repeatable)."
    ),
    max_iterations: int = typer.Option(5, "--max-iterations"),
    resume: str | None = typer.Option(
        None, "--resume", help="Resume a prior session id (under _meta/sessions/)."
    ),
    direction: str = typer.Option(
        "fwd",
        "--direction",
        help="Workflow direction: 'fwd' (sources→code, default) or 'rev' (code→sources).",
    ),
) -> None:
    """Open an interactive chat against the spec-writer agent."""
    state = cast(dict[str, object], ctx.obj or {})
    explicit_root = cast(Path | None, state.get("project_root"))

    try:
        project_ctx = (
            ProjectContext.from_cwd(explicit_root) if explicit_root else ProjectContext.from_cwd()
        )
    except NoProjectRootError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=11) from exc

    if resume:
        try:
            session = load_session(project_ctx, resume)
        except FileNotFoundError:
            typer.echo(f"session {resume!r} not found under _meta/sessions/", err=True)
            raise typer.Exit(code=12) from None
        typer.echo(f"resumed session {session.session_id} ({len(session.events)} events)")
    else:
        session = ChatSession(project_ctx=project_ctx)
        typer.echo(f"chat session {session.session_id} — type /help for commands, /quit to exit")

    # Best-effort initial methodology load — it's fine if missing; /reload
    # later will pick it up.
    initial_methodology: ResolvedMethodology | None = None
    try:
        initial_methodology = _load_methodology_for(project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(f"warning: methodology not yet loadable: {exc}", err=True)

    if direction not in ("fwd", "rev"):
        typer.echo(
            f"--direction must be 'fwd' or 'rev' (got {direction!r})",
            err=True,
        )
        raise typer.Exit(code=2)

    chat_agent: ChatAgent | None = None
    if initial_methodology is not None:
        chat_agent = _build_chat_agent_for_batch(
            project_ctx,
            initial_methodology,
            session,
            resume_from=resume,
            direction=direction,
        )

    cctx = ChatContext(
        project_ctx=project_ctx,
        session=session,
        default_layers=list(layer),
        max_iterations=max_iterations,
        methodology=initial_methodology,
        chat_agent=chat_agent,
    )
    session.log_note("chat started", layers=layer, resumed=bool(resume))

    set_answerer(_stdin_answerer(session))

    try:
        _repl(cctx)
    finally:
        set_answerer(None)
        session.log_note("chat ended")

    raise typer.Exit(code=0)


def _build_chat_agent_for_batch(
    project_ctx: ProjectContext,
    methodology: ResolvedMethodology,
    session: ChatSession,
    *,
    resume_from: str | None,
    direction: str = "fwd",
) -> ChatAgent | None:
    """Thin wrapper that translates :class:`UnsupportedProviderError`
    into a CLI-friendly warning + ``None`` return (preserving prior
    chat REPL behaviour)."""
    try:
        return build_chat_agent(
            project_ctx,
            methodology,
            session,
            direction=direction,
            resume_from=resume_from is not None,
        )
    except UnsupportedProviderError as exc:
        typer.echo(f"warning: {exc}; free-form input will be unavailable", err=True)
        return None


def _stdin_answerer(session: ChatSession) -> Callable[[str], str]:
    def _answer(question: str) -> str:
        typer.echo(f"[agent asks] {question}")
        line = sys.stdin.readline().rstrip("\n")
        session.log_note("ask_user", question=question, answer=line)
        return line

    return _answer


def _repl(cctx: ChatContext) -> None:
    """Read-eval-print loop. Returns when the user types /quit or stdin closes."""
    while True:
        try:
            line = input("» ").strip()
        except EOFError:
            return
        if not line:
            continue
        if line.startswith("/"):
            if _handle_slash(cctx, line) is False:
                return
        else:
            _handle_freeform(cctx, line)


def _handle_slash(cctx: ChatContext, line: str) -> bool:
    """Return False to signal the REPL should exit."""
    parts = shlex.split(line)
    command = parts[0].lower()
    args = parts[1:]
    cctx.session.log_slash(command, {"args": args} if args else None)

    handler = _SLASH_TABLE.get(command)
    if handler is None:
        typer.echo(f"unknown command {command!r}; try /help")
        return True

    if command in ("/quit", "/exit"):
        return False
    handler(cctx, args)
    return True


def _handle_freeform(cctx: ChatContext, text: str) -> None:
    """Send free-form input to the session-level ChatAgent.

    The agent decides what to do (read sources, list corpus, write
    specs, ask the user, …) — there is no fixed pipeline. Events are
    streamed to stdout as they arrive; ChatSession captures them in
    parallel for `--resume`.
    """
    if cctx.chat_agent is None:
        typer.echo(
            "chat agent unavailable (methodology not loaded or provider unsupported); "
            "use slash commands like /generate /refine for batch operations",
            err=True,
        )
        cctx.session.log_note("freeform skipped — no chat agent")
        return

    asyncio.run(_stream_chat_agent(cctx.chat_agent, text))


async def _stream_chat_agent(agent: ChatAgent, text: str) -> None:
    """Pump events from `agent.handle_user(text)` to stdout live."""
    async for event in agent.handle_user(text):
        if isinstance(event, TextDeltaEvent):
            typer.echo(event.text, nl=False)
        elif isinstance(event, ToolCallEvent):
            typer.echo(f"\n[{event.tool}({_brief_args(event.args)})]", nl=True)
        elif isinstance(event, ToolResultEvent):
            typer.echo(f"[→ {_brief_result(event.result)}]")
        elif isinstance(event, StatusEvent):
            typer.echo(f"\n[status: {event.phase}] {event.details}")
    # Newline after the streamed text block so the next prompt isn't glued.
    typer.echo("")


def _brief_args(args: dict[str, Any]) -> str:
    """One-line preview of tool args (truncate long values)."""
    items: list[str] = []
    for key, value in args.items():
        s = str(value)
        if len(s) > 80:
            s = s[:77] + "…"
        items.append(f"{key}={s}")
    joined = ", ".join(items)
    return joined if len(joined) <= 160 else joined[:157] + "…"


def _brief_result(result: object) -> str:
    text = str(result)
    return text if len(text) <= 120 else text[:117] + "…"


def _run_pipeline(
    cctx: ChatContext,
    *,
    direction: str,
    layers: list[str],
    mode: str,
    instruction: str | None,
    scope: Scope | None = None,
) -> None:
    try:
        result = run_pipeline_for(
            cctx.project_ctx,
            direction=direction,
            layers=layers,
            mode=mode,
            max_iterations=cctx.max_iterations,
            instruction=instruction,
            scope=scope,
        )
    except MethodologyLoadError as exc:
        typer.echo(f"methodology error: {exc}", err=True)
        cctx.session.log_note("methodology_error", error=str(exc))
        return
    except SystemExit:
        # run_pipeline_for itself shouldn't exit, but typer.Exit can travel
        cctx.session.log_note("pipeline_aborted")
        return

    cctx.session.log_note(
        "pipeline_completed",
        succeeded=result.succeeded,
        layers=[lr.layer_id for lr in result.layer_results],
    )
    try:
        _report_result(result)
    except typer.Exit:
        # _report_result raises Exit; absorb it inside chat
        pass


# ── slash handlers ──────────────────────────────────────────────────────


def _cmd_help(cctx: ChatContext, _args: list[str]) -> int:
    _ = cctx
    typer.echo("Commands:")
    for name in sorted(_SLASH_TABLE):
        typer.echo(f"  {name}")
    typer.echo("  (anything else) — free-form prompt → dialog-mode pipeline run")
    return 0


def _cmd_generate(cctx: ChatContext, args: list[str]) -> int:
    layers = _extract_layers(args) or cctx.default_layers
    if not layers:
        typer.echo("no layers — pass --layer L1 (repeatable)")
        return 0
    _run_pipeline(cctx, direction="fwd", layers=layers, mode="generate", instruction=None)
    return 0


def _cmd_refine(cctx: ChatContext, args: list[str]) -> int:
    layers = _extract_layers(args) or cctx.default_layers
    if not layers:
        typer.echo("no layers — pass --layer L1 (repeatable)")
        return 0
    _run_pipeline(cctx, direction="fwd", layers=layers, mode="refine", instruction=None)
    return 0


def _cmd_validate(cctx: ChatContext, _args: list[str]) -> int:
    from spec_writer.interface.batch.commands.generate import _load_methodology_for
    from spec_writer.io.spec_io import read_spec
    from spec_writer.methodology.validators.runner import run_validators
    from spec_writer.types import Spec

    try:
        methodology = _load_methodology_for(cctx.project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(f"methodology error: {exc}", err=True)
        return 1

    corpus: list[Spec] = []
    if cctx.project_ctx.specs.exists():
        for path in sorted(cctx.project_ctx.specs.rglob("*.md")):
            try:
                corpus.append(read_spec(path))
            except Exception:
                continue
    issues = run_validators(corpus, methodology)
    errors = [i for i in issues if i.severity == "error"]
    typer.echo(f"{len(errors)} error(s), {len(issues) - len(errors)} non-error(s)")
    for issue in errors[:10]:
        typer.echo(f"  [{issue.code}] {issue.spec_id or '?'}: {issue.message}")
    return 0


def _cmd_metrics(cctx: ChatContext, _args: list[str]) -> int:
    from spec_writer.interface.batch.commands.generate import (
        _load_existing_corpus,
        _load_methodology_for,
    )
    from spec_writer.methodology.metrics.runner import run_metrics

    try:
        methodology = _load_methodology_for(cctx.project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(f"methodology error: {exc}", err=True)
        return 1
    corpus = _load_existing_corpus(cctx.project_ctx)
    results = run_metrics(list(corpus.values()), methodology)
    if not results:
        typer.echo("(no metrics declared)")
        return 0
    for r in results:
        v = "n/a" if r.value is None else f"{r.value:.2f}"
        status = "ok"
        if r.error:
            status = f"ERROR ({r.error})"
        elif r.target_met is False:
            status = "FAIL" if r.critical else "below target"
        elif r.warning:
            status = f"WARN < {r.warn_below}"
        typer.echo(f"  {r.layer_id} {r.id}={v} {status}")
    return 0


def _cmd_status(cctx: ChatContext, _args: list[str]) -> int:
    specs_dir = cctx.project_ctx.specs
    n = sum(1 for _ in specs_dir.rglob("*.md")) if specs_dir.exists() else 0
    typer.echo(f"corpus: {n} spec(s) at {specs_dir}")
    typer.echo(f"session: {cctx.session.session_id}")
    typer.echo(f"default layers: {cctx.default_layers or '(none)'}")
    return 0


def _cmd_session(cctx: ChatContext, _args: list[str]) -> int:
    typer.echo(cctx.session.session_id)
    typer.echo(str(cctx.session.path))
    return 0


def _cmd_quit(_cctx: ChatContext, _args: list[str]) -> int:
    return 0


def _cmd_refine_plan(cctx: ChatContext, args: list[str]) -> int:
    """``/refine-plan [auto|errors|all]`` — preview validator findings, optionally apply.

    Without args: print the plan and prompt for a selection on stdin.
    With ``--apply <selection>``: skip the prompt, use the selection directly.
    Selection syntax mirrors :func:`select_plan_entries`.
    """
    apply_selection: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--apply" and i + 1 < len(args):
            apply_selection = args[i + 1]
            i += 2
        else:
            i += 1

    try:
        methodology = _load_methodology_for(cctx.project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(f"methodology error: {exc}", err=True)
        return 1

    corpus = _load_existing_corpus(cctx.project_ctx)
    plan = build_refine_plan(list(corpus.values()), methodology)
    typer.echo(plan.render())

    if not plan.entries:
        return 0

    if apply_selection is None:
        typer.echo(
            "\nSelect entries to refine (e.g. '1,3-5', 'errors', 'all', '' to skip): ",
            nl=False,
        )
        try:
            apply_selection = sys.stdin.readline().rstrip("\n")
        except EOFError:
            apply_selection = ""

    entries = select_plan_entries(plan, apply_selection or "")
    if not entries:
        typer.echo("no entries selected; nothing to refine")
        return 0

    ids = scope_spec_ids(entries)
    if not ids:
        # Cross-spec-only selection → fall back to full corpus refine
        typer.echo("only corpus-level issues selected — running full refine")
        layers = cctx.default_layers or sorted({s.layer for s in corpus.values()})
        _run_pipeline(cctx, direction="fwd", layers=layers, mode="refine", instruction=None)
        return 0

    layers = sorted({corpus[sid].layer for sid in ids if sid in corpus})
    typer.echo(f"refining {len(ids)} spec(s) across layers {layers}")
    _run_pipeline(
        cctx,
        direction="fwd",
        layers=layers,
        mode="refine",
        instruction=f"Refine plan selected {len(ids)} spec(s): {ids}",
        scope=Scope(spec_ids=ids),
    )
    return 0


def _cmd_cascade_fix(cctx: ChatContext, args: list[str]) -> int:
    """``/cascade-fix <spec_id>`` — propagate a manual edit through neighbours."""
    if not args:
        typer.echo("usage: /cascade-fix <spec_id> [--depth N]")
        return 0
    spec_id = args[0]
    depth = 2
    for i in range(1, len(args)):
        if args[i] == "--depth" and i + 1 < len(args):
            try:
                depth = max(1, int(args[i + 1]))
            except ValueError:
                pass

    corpus = _load_existing_corpus(cctx.project_ctx)
    if spec_id not in corpus:
        typer.echo(f"spec {spec_id!r} not found in corpus")
        return 1

    neighbours = cascade_neighbours(spec_id, corpus, depth=depth)
    typer.echo(f"cascade target ids ({len(neighbours)}): {', '.join(neighbours)}")

    layers = sorted({corpus[sid].layer for sid in neighbours if sid in corpus})
    scope = Scope(spec_ids=neighbours)
    _run_pipeline(
        cctx,
        direction="fwd",
        layers=layers,
        mode="cascade-fix",
        instruction=f"User just edited {spec_id}. Re-check the {len(neighbours)} related "
        f"specs and patch anything inconsistent. Do NOT touch specs outside this set.",
        scope=scope,
    )
    return 0


def _cmd_edit(cctx: ChatContext, args: list[str]) -> int:
    """``/edit <spec_id> <field>=<value> [<field>=<value> …]`` — patch + cascade."""
    if len(args) < 2 or "=" not in args[1]:
        typer.echo("usage: /edit <spec_id> field=value [field=value …]")
        return 0
    spec_id = args[0]
    changes: dict[str, Any] = {}
    for token in args[1:]:
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        changes[key.strip()] = value.strip()

    # Apply patch through the in-process tool (no LLM needed for the edit itself).
    import asyncio

    from spec_writer.agent.tool_registry import ToolContext
    from spec_writer.agent.tools.patch_spec import patch_spec
    from spec_writer.engine.id_registry import IdRegistry

    try:
        methodology = _load_methodology_for(cctx.project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(f"methodology error: {exc}", err=True)
        return 1
    id_registry = IdRegistry.from_disk(cctx.project_ctx.specs)
    tool_ctx = ToolContext(
        project_ctx=cctx.project_ctx, methodology=methodology, id_registry=id_registry
    )
    try:
        result = asyncio.run(patch_spec({"spec_id": spec_id, "changes": changes}, tool_ctx))
    except Exception as exc:
        typer.echo(f"patch failed: {exc}", err=True)
        cctx.session.log_note("edit_failed", spec_id=spec_id, error=str(exc))
        return 1
    if not result.get("ok"):
        typer.echo(f"patch failed: {result.get('error')}", err=True)
        return 1
    typer.echo(f"patched {spec_id}: {sorted(changes)}")
    cctx.session.log_note("edit_applied", spec_id=spec_id, changes=sorted(changes))

    # Trigger targeted-edit-cascade against neighbours
    corpus = _load_existing_corpus(cctx.project_ctx)
    neighbours = cascade_neighbours(spec_id, corpus, depth=2)
    layers = sorted({corpus[sid].layer for sid in neighbours if sid in corpus})
    if not layers:
        typer.echo("(no neighbours to cascade)")
        return 0
    scope = Scope(spec_ids=neighbours)
    _run_pipeline(
        cctx,
        direction="fwd",
        layers=layers,
        mode="targeted-edit-cascade",
        instruction=f"Spec {spec_id} was just edited with: {changes}. "
        f"Cascade the implications across the {len(neighbours) - 1} related specs.",
        scope=scope,
    )
    return 0


def _cmd_add_feature(cctx: ChatContext, args: list[str]) -> int:
    """``/add-feature "<description>"`` — run pipeline in incremental-add mode.

    The remainder of the command line (after layers and other flags) is
    joined as the natural-language description and passed in as the
    agent's ``--instruction``. Use ``--layer`` (repeatable) to scope.
    """
    layers = _extract_layers(args) or cctx.default_layers
    description_parts: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--layer" and i + 1 < len(args):
            i += 2
            continue
        description_parts.append(args[i])
        i += 1
    description = " ".join(description_parts).strip()
    if not description:
        typer.echo('usage: /add-feature [--layer L2 …] "<feature description>"')
        return 0
    if not layers:
        typer.echo("no layers — pass --layer L2 (repeatable) or start chat with --layer")
        return 0
    _run_pipeline(
        cctx,
        direction="fwd",
        layers=layers,
        mode="incremental-add",
        instruction=description,
    )
    return 0


def _cmd_reload(cctx: ChatContext, _args: list[str]) -> int:
    """Re-read methodology.yaml + mixins; print a diff vs. the previous load."""
    try:
        new_m = _load_methodology_for(cctx.project_ctx)
    except MethodologyLoadError as exc:
        typer.echo(f"methodology error: {exc}", err=True)
        cctx.session.log_note("reload_failed", error=str(exc))
        return 1

    if cctx.methodology is None:
        typer.echo("methodology loaded (first time this session)")
        cctx.methodology = new_m
        cctx.session.log_note("methodology_loaded")
        return 0

    diff = diff_methodologies(cctx.methodology, new_m)
    cctx.methodology = new_m
    rendered = diff.render()
    typer.echo(rendered)
    cctx.session.log_note("methodology_reloaded", diff=rendered)
    return 0


def _extract_layers(args: list[str]) -> list[str]:
    layers: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--layer" and i + 1 < len(args):
            layers.append(args[i + 1])
            i += 2
        else:
            i += 1
    return layers


_SLASH_TABLE: dict[str, Callable[[ChatContext, list[str]], int]] = {
    "/help": _cmd_help,
    "/generate": _cmd_generate,
    "/refine": _cmd_refine,
    "/add-feature": _cmd_add_feature,
    "/refine-plan": _cmd_refine_plan,
    "/cascade-fix": _cmd_cascade_fix,
    "/edit": _cmd_edit,
    "/validate": _cmd_validate,
    "/metrics": _cmd_metrics,
    "/status": _cmd_status,
    "/session": _cmd_session,
    "/reload": _cmd_reload,
    "/quit": _cmd_quit,
    "/exit": _cmd_quit,
}


def _slash_table_for_tests() -> dict[str, Any]:
    return dict(_SLASH_TABLE)


__all__ = ["ChatContext", "run"]
