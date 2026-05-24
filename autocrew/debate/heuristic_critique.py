"""Rule-based agent critiques when no LLM API keys are configured."""

from __future__ import annotations

from autocrew.analyzer.project_model import ProjectContext
from autocrew.debate.debate_model import AgentCritique
from autocrew.squad.squad_model import AgentConfig, AgentRole


def _features_by_status(context: ProjectContext, *statuses: str) -> list[str]:
    return [f.name for f in context.features if f.status in statuses]


def _high_priority_gaps(context: ProjectContext) -> list[str]:
    gaps = _features_by_status(context, "not_started", "partial")
    return [
        f.name
        for f in context.features
        if f.name in gaps and f.priority in ("high", "medium")
    ]


def _plan_covers(plan_text: str, keywords: list[str]) -> bool:
    lower = plan_text.lower()
    return any(kw.lower() in lower for kw in keywords)


def generate_heuristic_critique(
    agent: AgentConfig,
    context: ProjectContext,
    plan_text: str,
    round_number: int,
) -> AgentCritique:
    concerns: list[str] = []
    suggestions: list[str] = []
    blockers: list[str] = []

    missing = _features_by_status(context, "not_started")
    partial = _features_by_status(context, "partial")
    high_gaps = _high_priority_gaps(context)

    role = agent.role

    if role == AgentRole.PRODUCT_OWNER:
        if missing:
            concerns.append(f"{len(missing)} features still not started: {', '.join(missing[:5])}")
        if partial:
            concerns.append(f"{len(partial)} features only partial: {', '.join(partial[:5])}")
        if context.missing_parts:
            for part in context.missing_parts[:5]:
                if not _plan_covers(plan_text, [part]):
                    suggestions.append(f"Add acceptance criteria for: {part}")
        if high_gaps and not _plan_covers(plan_text, high_gaps):
            blockers.append(f"Plan must include high-priority items: {', '.join(high_gaps[:3])}")

    elif role == AgentRole.ARCHITECT:
        stack = context.tech_stack.frontend + context.tech_stack.backend
        if stack and not _plan_covers(plan_text, stack[:3]):
            concerns.append("Architecture doc should reference core stack components explicitly")
        if context.domain.value == "mobile_app" and "mobile" not in plan_text.lower():
            suggestions.append("Document mobile app module boundaries (Expo Router, features/)")
        if high_gaps:
            suggestions.append("Define service boundaries before implementing: " + ", ".join(high_gaps[:3]))

    elif role == AgentRole.BACKEND_DEV:
        backend_gaps = [n for n in high_gaps if any(k in n.lower() for k in ("api", "auth", "payment", "review", "notification", "job", "admin"))]
        if not backend_gaps:
            backend_gaps = high_gaps
        for name in backend_gaps[:4]:
            if not _plan_covers(plan_text, [name, "api", "nestjs"]):
                blockers.append(f"Backend plan missing for: {name}")
        if "payment" in " ".join(missing + partial).lower() and "stripe" not in plan_text.lower():
            blockers.append("Payment integration needs Stripe webhook + checkout API design")

    elif role == AgentRole.FRONTEND_DEV:
        ui_gaps = [n for n in high_gaps if any(k in n.lower() for k in ("dashboard", "portal", "profile", "notification", "payment", "mobile"))]
        for name in ui_gaps[:4]:
            if not _plan_covers(plan_text, [name, "screen", "mobile", "ui"]):
                concerns.append(f"Frontend screens not planned for: {name}")
        if "admin" in " ".join(missing).lower() and "next.js" not in plan_text.lower():
            suggestions.append("Admin dashboard should specify Next.js app location in monorepo")

    elif role == AgentRole.DEVOPS:
        if context.tech_stack.devops and not _plan_covers(plan_text, context.tech_stack.devops):
            concerns.append("DevOps stack not reflected in deployment plan")
        if high_gaps and "ci" not in plan_text.lower() and "deploy" not in plan_text.lower():
            suggestions.append("Add CI/CD gates for new features before merge")

    elif role == AgentRole.DATA_ENGINEER:
        data_keywords = ("migration", "schema", "database", "prisma", "postgis")
        if any(k in context.description.lower() for k in data_keywords):
            if not _plan_covers(plan_text, list(data_keywords)):
                concerns.append("Data layer / Prisma migrations should be in the plan")
        for name in high_gaps:
            if any(k in name.lower() for k in ("payment", "review", "notification")):
                suggestions.append(f"Confirm DB schema changes needed for: {name}")

    elif role == AgentRole.AI_ENGINEER:
        if context.domain.value == "ai_tool" or "llm" in context.description.lower():
            if not _plan_covers(plan_text, ["ai", "llm", "embedding"]):
                concerns.append("AI features need explicit integration plan")
        else:
            suggestions.append("No AI scope required for this MVP — approved from AI perspective")

    elif role == AgentRole.TESTER:
        if high_gaps:
            blockers.append(f"Cannot approve without test plan for: {', '.join(high_gaps[:3])}")
        if not _plan_covers(plan_text, ["test", "jest", "e2e"]):
            suggestions.append("Add test coverage targets for partial and new features")

    elif role == AgentRole.CODE_REVIEWER:
        if any(k in " ".join(high_gaps).lower() for k in ("payment", "auth")):
            if not _plan_covers(plan_text, ["security", "review", "error handling"]):
                concerns.append("Security review criteria missing for auth/payment features")
        if partial:
            suggestions.append(f"Review stub/TODO code in partial features: {', '.join(partial[:3])}")

    elif role == AgentRole.PROGRESS_TRACKER:
        done_count = len(_features_by_status(context, "done"))
        total = len(context.features) or 1
        pct = done_count / total * 100
        if pct < 100 and not _plan_covers(plan_text, ["priority", "next"]):
            suggestions.append(f"Plan should prioritize remaining {100 - pct:.0f}% of work")
        if context.missing_parts:
            for part in context.missing_parts[:3]:
                if not _plan_covers(plan_text, [part.split()[0]]):
                    concerns.append(f"Tracker: missing part not in plan — {part}")

    elif role == AgentRole.FULLSTACK_DEV:
        for name in high_gaps[:3]:
            if not _plan_covers(plan_text, [name]):
                concerns.append(f"End-to-end implementation plan needed for: {name}")

    approved = len(blockers) == 0 and len(concerns) == 0

    return AgentCritique(
        agent_role=agent.role.value,
        agent_name=agent.name,
        round_number=round_number,
        approved=approved,
        concerns=concerns,
        suggestions=suggestions,
        blockers=blockers,
    )
