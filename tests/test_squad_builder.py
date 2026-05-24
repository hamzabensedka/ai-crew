from autocrew.analyzer.project_model import (
    FeatureItem,
    ProjectContext,
    ProjectDomain,
    ProjectType,
    TechStack,
)
from autocrew.squad.squad_builder import build_squad
from autocrew.squad.squad_model import AgentRole


def _context(**kwargs) -> ProjectContext:
    defaults = dict(
        project_type=ProjectType.NEW_IDEA,
        project_name="TestApp",
        domain=ProjectDomain.SAAS,
        description="A test app",
        tech_stack=TechStack(),
    )
    defaults.update(kwargs)
    return ProjectContext(**defaults)


class TestRoleSelection:
    def test_always_includes_core_roles(self):
        squad = build_squad(_context())
        roles = {a.role for a in squad.agents}
        assert AgentRole.PRODUCT_OWNER in roles
        assert AgentRole.ARCHITECT in roles
        assert AgentRole.CODE_REVIEWER in roles
        assert AgentRole.PROGRESS_TRACKER in roles

    def test_includes_frontend_when_stack_has_frontend(self):
        features = [FeatureItem(name=f"F{i}", description="d") for i in range(6)]
        squad = build_squad(
            _context(
                tech_stack=TechStack(frontend=["Next.js"], backend=["FastAPI"]),
                features=features,
            )
        )
        roles = {a.role for a in squad.agents}
        assert AgentRole.FRONTEND_DEV in roles
        assert AgentRole.BACKEND_DEV in roles

    def test_fullstack_for_small_projects(self):
        features = [FeatureItem(name=f"F{i}", description="d") for i in range(3)]
        squad = build_squad(
            _context(
                tech_stack=TechStack(frontend=["React"], backend=["FastAPI"]),
                features=features,
            )
        )
        roles = {a.role for a in squad.agents}
        assert AgentRole.FULLSTACK_DEV in roles

    def test_tester_for_many_features(self):
        features = [FeatureItem(name=f"F{i}", description="d") for i in range(7)]
        squad = build_squad(_context(features=features))
        roles = {a.role for a in squad.agents}
        assert AgentRole.TESTER in roles

    def test_ai_engineer_for_ai_domain(self):
        squad = build_squad(_context(domain=ProjectDomain.AI_TOOL))
        roles = {a.role for a in squad.agents}
        assert AgentRole.AI_ENGINEER in roles

    def test_devops_when_devops_stack(self):
        squad = build_squad(_context(tech_stack=TechStack(devops=["Docker"])))
        roles = {a.role for a in squad.agents}
        assert AgentRole.DEVOPS in roles


class TestExecutionPlan:
    def test_po_and_architect_first(self):
        squad = build_squad(_context())
        assert squad.execution_order[0] == AgentRole.PRODUCT_OWNER.value
        assert squad.execution_order[1] == AgentRole.ARCHITECT.value

    def test_tracker_near_end(self):
        squad = build_squad(
            _context(tech_stack=TechStack(frontend=["React"], backend=["FastAPI"]))
        )
        assert AgentRole.PROGRESS_TRACKER.value in squad.execution_order
        tracker_idx = squad.execution_order.index(AgentRole.PROGRESS_TRACKER.value)
        assert tracker_idx > 1

    def test_parallel_groups_for_multiple_devs(self):
        squad = build_squad(
            _context(
                tech_stack=TechStack(frontend=["React"], backend=["FastAPI"]),
                features=[FeatureItem(name=f"F{i}", description="d") for i in range(7)],
            )
        )
        assert len(squad.parallel_groups) >= 1


class TestSquadSerialization:
    def test_roundtrip(self):
        squad = build_squad(_context())
        restored = type(squad).from_dict(squad.to_dict())
        assert restored.project_name == squad.project_name
        assert len(restored.agents) == len(squad.agents)
