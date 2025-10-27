"""
Design Problem Scenarios

This module defines various design problems for testing collective critique
reasoning. Each problem represents a realistic design challenge that benefits
from multiple expert perspectives.

Design problem types include:
- UI/UX design
- API design
- System architecture
- Data models
- User workflows
- Information architecture
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DesignDomain(Enum):
    """Types of design domains"""
    UI_UX = "ui_ux"  # User interface and experience
    API = "api"  # API and interface design
    SYSTEM = "system"  # System architecture
    DATA = "data"  # Data models and schemas
    WORKFLOW = "workflow"  # User workflows and processes
    INFORMATION = "information"  # Information architecture
    INTERACTION = "interaction"  # Interaction patterns


class CritiquePerspective(Enum):
    """Different critique perspectives"""
    USABILITY = "usability"  # User experience and ease of use
    ACCESSIBILITY = "accessibility"  # Inclusive design
    PERFORMANCE = "performance"  # Efficiency and speed
    SECURITY = "security"  # Safety and privacy
    MAINTAINABILITY = "maintainability"  # Long-term sustainability
    SCALABILITY = "scalability"  # Growth and expansion
    AESTHETICS = "aesthetics"  # Visual and conceptual appeal
    CONSISTENCY = "consistency"  # Internal coherence
    USER_ADVOCACY = "user_advocacy"  # User needs and goals


@dataclass
class DesignProblem:
    """A design problem that needs critique"""
    name: str
    domain: DesignDomain
    description: str
    current_design: str  # The design to critique
    context: str  # Background and constraints
    success_criteria: List[str]  # What makes a good solution
    known_issues: List[str] = field(default_factory=list)  # Optional known problems
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_critique_prompt(
        self,
        perspective: Optional[CritiquePerspective] = None
    ) -> str:
        """Convert to a prompt for design critique"""
        prompt_parts = []

        if perspective:
            prompt_parts.append(
                f"You are a {perspective.value} expert reviewing this design.\n"
            )
        else:
            prompt_parts.append("Review this design and provide constructive critique.\n")

        prompt_parts.append(f"DESIGN PROBLEM: {self.description}\n")
        prompt_parts.append(f"\nCONTEXT:\n{self.context}\n")
        prompt_parts.append(f"\nCURRENT DESIGN:\n{self.current_design}\n")

        prompt_parts.append(f"\nSUCCESS CRITERIA:")
        for i, criterion in enumerate(self.success_criteria, 1):
            prompt_parts.append(f"{i}. {criterion}")

        if perspective:
            prompt_parts.append(
                f"\nProvide critique from a {perspective.value} perspective. "
                "Consider:\n"
                "1. What works well?\n"
                "2. What are potential issues or risks?\n"
                "3. What specific improvements would you recommend?\n"
            )
        else:
            prompt_parts.append(
                "\nProvide comprehensive critique. Consider:\n"
                "1. Strengths of the current design\n"
                "2. Weaknesses or potential issues\n"
                "3. Specific, actionable recommendations\n"
            )

        return "\n".join(prompt_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "name": self.name,
            "domain": self.domain.value,
            "description": self.description,
            "current_design": self.current_design,
            "context": self.context,
            "success_criteria": self.success_criteria,
            "known_issues": self.known_issues,
            "difficulty": self.difficulty,
            "metadata": self.metadata
        }


# ============================================================================
# UI/UX Design Problems
# ============================================================================

MOBILE_CHECKOUT = DesignProblem(
    name="mobile_checkout_flow",
    domain=DesignDomain.UI_UX,
    description="Design a mobile checkout flow for an e-commerce app",
    current_design="""
SCREEN 1: Cart Summary
- List of items with thumbnails
- Quantity adjusters (+/-)
- Remove item buttons
- "Proceed to Checkout" button at bottom

SCREEN 2: Shipping Address
- Full form: Name, Address Line 1, Address Line 2, City, State, ZIP
- "Save address for future" checkbox
- "Continue" button

SCREEN 3: Payment Method
- Credit card form: Number, Expiry, CVV, Billing ZIP
- "Save card for future" checkbox
- "Review Order" button

SCREEN 4: Order Review
- Summary of items, shipping, payment
- "Place Order" button
""",
    context="""
- Mobile-first e-commerce app for fashion/apparel
- Target users: 25-45 year olds, varying tech literacy
- Average cart value: $80
- Current conversion rate: 62% (industry average: 70%)
- Users report checkout feels "too long"
- App needs to work on iOS and Android
""",
    success_criteria=[
        "High conversion rate (minimize cart abandonment)",
        "Fast checkout for returning users",
        "Clear, unambiguous flow",
        "Accessible to users with disabilities",
        "Secure handling of sensitive data",
        "Mobile-optimized interactions"
    ],
    known_issues=[
        "Current flow is 4 screens, users say it's too long",
        "Form fields are small on mobile devices",
        "No guest checkout option",
        "Error messages appear after form submission, not inline"
    ],
    difficulty="medium"
)

DASHBOARD_LAYOUT = DesignProblem(
    name="analytics_dashboard_layout",
    domain=DesignDomain.UI_UX,
    description="Design a layout for a data analytics dashboard",
    current_design="""
LAYOUT:
[Navigation Sidebar - 200px fixed width]
[Main Content Area]
  - Page Title
  - 4 KPI cards in a row (Revenue, Users, Conversion, Engagement)
  - Large line chart showing trend over time (full width)
  - Data table below chart (full width, 10 rows visible, pagination)
  - Export button (top right)
""",
    context="""
- B2B SaaS analytics platform
- Users: Marketing managers, data analysts, executives
- Users need to monitor metrics and identify trends
- Typical session: 10-20 minutes, checking multiple metrics
- Desktop-focused (90% desktop usage)
- Users often compare time periods or segments
- Real-time data updates every 60 seconds
""",
    success_criteria=[
        "Quick insight discovery (key metrics visible immediately)",
        "Efficient comparison of data across dimensions",
        "Customizable for different user roles",
        "Clear visual hierarchy",
        "Performant with large datasets"
    ],
    known_issues=[
        "Users can't customize which KPIs appear",
        "No way to filter or segment data without leaving page",
        "Chart doesn't support comparison mode",
        "No drill-down capability from high-level metrics"
    ],
    difficulty="medium"
)

# ============================================================================
# API Design Problems
# ============================================================================

REST_API_VERSIONING = DesignProblem(
    name="rest_api_versioning",
    domain=DesignDomain.API,
    description="Design an API versioning strategy for a growing platform",
    current_design="""
CURRENT APPROACH:
- Version in URL path: /api/v1/users, /api/v2/users
- Breaking changes trigger new major version
- Old versions deprecated after 6 months, removed after 12 months
- No version negotiation
- No per-endpoint versioning

EXAMPLE ENDPOINTS:
GET /api/v1/users/{id}
POST /api/v1/users
PUT /api/v1/users/{id}
DELETE /api/v1/users/{id}

PLANNED BREAKING CHANGE:
- Changing user response format (renaming fields, restructuring nested data)
- Adding required fields to POST request
- Changing date format from timestamps to ISO 8601
""",
    context="""
- Public API with 500+ external clients
- Mobile apps, web apps, third-party integrations
- Some clients are slow to update (enterprises on 12+ month cycles)
- Team wants to move fast but not break clients
- Currently only 2 versions active (v1, v2)
- API serves 10M requests/day
- Need to support gradual migration
""",
    success_criteria=[
        "Allow API evolution without breaking existing clients",
        "Clear migration path for clients",
        "Minimal maintenance burden for multiple versions",
        "Discoverable and well-documented",
        "Support both rapid iteration and stability"
    ],
    difficulty="hard"
)

GRAPHQL_SCHEMA = DesignProblem(
    name="graphql_schema_design",
    domain=DesignDomain.API,
    description="Design a GraphQL schema for a social media platform",
    current_design="""
type User {
  id: ID!
  username: String!
  email: String!
  posts: [Post!]!
  followers: [User!]!
  following: [User!]!
  createdAt: String!
}

type Post {
  id: ID!
  author: User!
  content: String!
  likes: [User!]!
  comments: [Comment!]!
  createdAt: String!
}

type Comment {
  id: ID!
  author: User!
  post: Post!
  content: String!
  createdAt: String!
}

type Query {
  user(id: ID!): User
  post(id: ID!): Post
  feed: [Post!]!
}
""",
    context="""
- Social media app with posts, comments, likes, follows
- Expected scale: 1M users, 10M posts
- Feed is personalized and paginated
- Users can have thousands of followers/following
- Privacy settings: public, friends-only, private posts
- Need to support efficient feed generation
- Want to avoid N+1 query problems
""",
    success_criteria=[
        "Efficient data fetching (avoid over-fetching and under-fetching)",
        "Scalable for large lists (followers, posts, etc.)",
        "Flexible enough for different client needs",
        "Respect privacy and permissions",
        "Good developer experience"
    ],
    known_issues=[
        "No pagination on lists",
        "No privacy/permissions model",
        "followers/following could be huge arrays",
        "feed query has no filtering options",
        "Potential for expensive nested queries"
    ],
    difficulty="hard"
)

# ============================================================================
# System Architecture Problems
# ============================================================================

MICROSERVICES_SPLIT = DesignProblem(
    name="microservices_split_strategy",
    domain=DesignDomain.SYSTEM,
    description="Design a strategy to split a monolith into microservices",
    current_design="""
CURRENT MONOLITH:
- Single Ruby on Rails application
- PostgreSQL database with 150 tables
- Key modules:
  * User management (auth, profiles, permissions)
  * Product catalog (items, categories, inventory)
  * Order processing (cart, checkout, fulfillment)
  * Payment handling (transactions, refunds)
  * Notifications (email, SMS, push)
  * Analytics (tracking, reporting)

PROPOSED SPLIT:
Service 1: User Service (auth, profiles, permissions)
Service 2: Catalog Service (products, inventory)
Service 3: Order Service (cart, checkout, orders)
Service 4: Payment Service (transactions)
Service 5: Notification Service (all notifications)

COMMUNICATION:
- REST APIs between services
- Shared PostgreSQL database (separate schemas per service)
- No message queue yet
""",
    context="""
- Current monolith: 200k LOC, 15 developers
- Pain points: slow deploys, tight coupling, hard to scale
- Team organized around features, not services
- Need to maintain system while migrating
- Can't afford "big bang" rewrite
- Some tables are used by multiple domains
- Transaction consistency is critical for orders/payments
""",
    success_criteria=[
        "Independent deployment of services",
        "Clear service boundaries and responsibilities",
        "Maintain data consistency",
        "Minimal downtime during migration",
        "Support team autonomy",
        "Scalable architecture"
    ],
    known_issues=[
        "Shared database creates tight coupling",
        "No clear strategy for cross-service transactions",
        "Notification service seems too generic",
        "No plan for handling service failures",
        "Data migration strategy unclear"
    ],
    difficulty="hard"
)

CACHING_STRATEGY = DesignProblem(
    name="multi_layer_caching",
    domain=DesignDomain.SYSTEM,
    description="Design a caching strategy for a high-traffic content platform",
    current_design="""
CURRENT APPROACH:
- Redis cache for database query results
- TTL: 5 minutes for all cached items
- Cache key format: "table:id" (e.g., "posts:123")
- Cache on read (lazy loading)
- Full cache invalidation on any write to a record

CONTENT TYPES:
- User profiles (change infrequently)
- Posts (created often, rarely updated)
- Comments (created often, never updated)
- Likes/reactions (high write volume)
- Feed (personalized, complex query)

CURRENT ISSUES:
- Cache hit rate: 45% (goal: 80%+)
- Invalidation causes cascading misses
- Same TTL for all content types
- No CDN layer
""",
    context="""
- Content platform with 500k daily active users
- Read:write ratio approximately 100:1
- Most reads are recent content (last 24 hours)
- Users refresh feeds frequently
- Some content is popular (viral posts)
- Latency target: p95 < 200ms
- Need to balance freshness and performance
""",
    success_criteria=[
        "High cache hit rate (80%+)",
        "Low latency for common operations",
        "Fresh content (no stale data shown)",
        "Efficient invalidation strategy",
        "Cost-effective (optimize Redis usage)",
        "Scalable to 10x traffic"
    ],
    difficulty="medium"
)

# ============================================================================
# Data Model Problems
# ============================================================================

PERMISSION_MODEL = DesignProblem(
    name="flexible_permission_system",
    domain=DesignDomain.DATA,
    description="Design a flexible permission and access control system",
    current_design="""
DATABASE SCHEMA:

users
- id
- email
- role (enum: admin, manager, member, guest)

resources
- id
- type (enum: document, project, folder)
- owner_id (FK to users)
- created_at

permissions
- id
- user_id (FK to users)
- resource_id (FK to resources)
- can_read (boolean)
- can_write (boolean)
- can_delete (boolean)
- can_share (boolean)

ACCESS LOGIC:
- Owner has all permissions
- Admins have all permissions on everything
- Others only have permissions via permissions table
- No group/team concept
- No inheritance (folder permissions don't cascade)
""",
    context="""
- Collaborative workspace application
- Need to support teams, projects, folders with nested resources
- Use cases:
  * Share document with specific people
  * Give team access to project
  * Inherit folder permissions to contents
  * Delegate admin rights for specific projects
  * Temporary access (expire after date)
- Current model is too rigid and doesn't scale
- Querying permissions is becoming slow (millions of permission records)
""",
    success_criteria=[
        "Support complex permission scenarios",
        "Efficient permission checks (fast queries)",
        "Support teams/groups, not just individual users",
        "Permission inheritance for nested resources",
        "Auditable (who has access to what)",
        "Easy to reason about and debug"
    ],
    known_issues=[
        "No team/group concept",
        "No permission inheritance",
        "No time-based permissions",
        "Permission queries are slow",
        "Hard to answer 'who has access to X'"
    ],
    difficulty="hard"
)

# ============================================================================
# Workflow Design Problems
# ============================================================================

APPROVAL_WORKFLOW = DesignProblem(
    name="multi_stage_approval",
    domain=DesignDomain.WORKFLOW,
    description="Design a flexible multi-stage approval workflow",
    current_design="""
CURRENT WORKFLOW:
1. User creates request
2. Request sent to manager
3. Manager approves/rejects
4. If approved, request is processed
5. User notified of outcome

WORKFLOW DATA:
- request_id, user_id, manager_id
- status (pending, approved, rejected, processed)
- created_at, updated_at
- Hard-coded in application logic

LIMITATIONS:
- Only supports single approver
- Can't skip or reroute
- No conditional logic
- No parallel approvals
- Can't add ad-hoc approvers
""",
    context="""
- Enterprise expense/purchase approval system
- Different request types need different flows:
  * Small expenses: 1 approver
  * Large expenses: manager → director → finance
  * Technical purchases: manager → IT → procurement
  * Some need parallel approvals (budget + technical)
- Approvers may delegate or escalate
- Need audit trail
- Business users want to configure flows without engineering
""",
    success_criteria=[
        "Support multiple approval stages",
        "Flexible routing (conditional, parallel, sequential)",
        "Configurable by business users",
        "Handle delegation and escalation",
        "Complete audit trail",
        "Easy to understand and debug"
    ],
    difficulty="hard"
)

# ============================================================================
# Registry
# ============================================================================

ALL_PROBLEMS = [
    MOBILE_CHECKOUT,
    DASHBOARD_LAYOUT,
    REST_API_VERSIONING,
    GRAPHQL_SCHEMA,
    MICROSERVICES_SPLIT,
    CACHING_STRATEGY,
    PERMISSION_MODEL,
    APPROVAL_WORKFLOW,
]

PROBLEMS_BY_DOMAIN = {
    DesignDomain.UI_UX: [MOBILE_CHECKOUT, DASHBOARD_LAYOUT],
    DesignDomain.API: [REST_API_VERSIONING, GRAPHQL_SCHEMA],
    DesignDomain.SYSTEM: [MICROSERVICES_SPLIT, CACHING_STRATEGY],
    DesignDomain.DATA: [PERMISSION_MODEL],
    DesignDomain.WORKFLOW: [APPROVAL_WORKFLOW],
}

PROBLEMS_BY_DIFFICULTY = {
    "easy": [p for p in ALL_PROBLEMS if p.difficulty == "easy"],
    "medium": [p for p in ALL_PROBLEMS if p.difficulty == "medium"],
    "hard": [p for p in ALL_PROBLEMS if p.difficulty == "hard"],
}

PROBLEMS_BY_NAME = {p.name: p for p in ALL_PROBLEMS}


def get_problem(name: str) -> Optional[DesignProblem]:
    """Get a problem by name"""
    return PROBLEMS_BY_NAME.get(name)


def get_problems_by_domain(domain: DesignDomain) -> List[DesignProblem]:
    """Get all problems in a specific domain"""
    return PROBLEMS_BY_DOMAIN.get(domain, [])


def get_problems_by_difficulty(difficulty: str) -> List[DesignProblem]:
    """Get all problems of a specific difficulty"""
    return PROBLEMS_BY_DIFFICULTY.get(difficulty, [])
