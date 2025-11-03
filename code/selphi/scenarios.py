"""
Theory of Mind Scenarios

This module defines various scenarios for testing theory of mind and epistemology
in language models. Each scenario tests specific aspects of:
- False belief understanding
- Knowledge attribution
- Perspective taking
- Belief updating
- Second-order beliefs

Scenarios are structured with:
- setup: The initial situation
- events: What happens (may change beliefs/knowledge)
- test_questions: Questions to probe understanding
- correct_answers: Expected responses
- reasoning: Why this tests ToM
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ToMType(Enum):
    """Types of Theory of Mind tests"""

    FALSE_BELIEF = "false_belief"  # Classic Sally-Anne style
    KNOWLEDGE_ATTRIBUTION = "knowledge_attribution"  # Who knows what
    PERSPECTIVE_TAKING = "perspective_taking"  # Different viewpoints
    BELIEF_UPDATING = "belief_updating"  # How beliefs change
    SECOND_ORDER_BELIEF = "second_order_belief"  # Beliefs about beliefs
    EPISTEMIC_STATE = "epistemic_state"  # Knowing vs believing vs guessing
    PRAGMATIC_REASONING = "pragmatic_reasoning"  # Implied knowledge/intent


@dataclass
class ToMScenario:
    """A theory of mind test scenario"""

    name: str
    tom_type: ToMType
    setup: str
    events: List[str]
    test_questions: List[str]
    correct_answers: List[str]
    reasoning: str
    difficulty: str = "medium"  # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_prompt(self, include_context: bool = True) -> str:
        """Convert scenario to a prompt for the model"""
        prompt_parts = []

        if include_context:
            prompt_parts.append("Read the following scenario carefully and answer the questions.\n")

        prompt_parts.append(f"Setup: {self.setup}\n")

        if self.events:
            prompt_parts.append("\nEvents:")
            for i, event in enumerate(self.events, 1):
                prompt_parts.append(f"{i}. {event}")
            prompt_parts.append("")

        prompt_parts.append("\nQuestions:")
        for i, question in enumerate(self.test_questions, 1):
            prompt_parts.append(f"{i}. {question}")

        return "\n".join(prompt_parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "name": self.name,
            "tom_type": self.tom_type.value,
            "setup": self.setup,
            "events": self.events,
            "test_questions": self.test_questions,
            "correct_answers": self.correct_answers,
            "reasoning": self.reasoning,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }


# ============================================================================
# Classic False Belief Scenarios
# ============================================================================

SALLY_ANNE = ToMScenario(
    name="sally_anne_classic",
    tom_type=ToMType.FALSE_BELIEF,
    setup="Sally has a basket and Anne has a box. Sally puts a marble in her basket and leaves the room.",
    events=[
        "While Sally is away, Anne takes the marble from Sally's basket and puts it in her box.",
        "Sally returns to the room.",
    ],
    test_questions=[
        "Where will Sally look for her marble?",
        "Where is the marble actually located?",
        "Does Sally know where the marble really is?",
    ],
    correct_answers=[
        "Sally will look in her basket (where she left it)",
        "The marble is in Anne's box",
        "No, Sally doesn't know Anne moved it",
    ],
    reasoning="Tests if the model understands that Sally has a false belief about the marble's location because she didn't observe Anne moving it.",
    difficulty="easy",
)

CHOCOLATE_BAR = ToMScenario(
    name="chocolate_bar_location",
    tom_type=ToMType.FALSE_BELIEF,
    setup="Emma puts a chocolate bar in the kitchen cupboard and goes outside to play.",
    events=[
        "While Emma is outside, her brother moves the chocolate bar from the cupboard to the refrigerator.",
        "Emma comes back inside, hungry for a snack.",
    ],
    test_questions=[
        "Where will Emma look for the chocolate bar?",
        "Why will she look there?",
        "Will she find it on her first try?",
    ],
    correct_answers=[
        "Emma will look in the kitchen cupboard",
        "Because that's where she left it and she doesn't know it was moved",
        "No, she won't find it there",
    ],
    reasoning="Tests false belief understanding in a everyday scenario with clear motivation.",
    difficulty="easy",
)

# ============================================================================
# Knowledge Attribution Scenarios
# ============================================================================

SURPRISE_PARTY = ToMScenario(
    name="surprise_party_knowledge",
    tom_type=ToMType.KNOWLEDGE_ATTRIBUTION,
    setup="Alex's friends are planning a surprise birthday party for him. They discuss the plans in a group chat that Alex is not part of.",
    events=[
        "Maria tells Tom about the party in person.",
        "Tom accidentally mentions 'the party' in front of Alex but doesn't say what it's for.",
        "Sarah, who is in the group chat, hasn't talked to anyone yet.",
    ],
    test_questions=[
        "Does Alex know there's a party being planned?",
        "Does Alex know it's for his birthday?",
        "Does Maria know about the party?",
        "Does Sarah know about the party?",
        "Who has the most complete knowledge about the party?",
    ],
    correct_answers=[
        "Alex might know there's some party, but not that it's for him",
        "No, Alex doesn't know it's for his birthday",
        "Yes, Maria knows about the party",
        "Yes, Sarah knows about the party",
        "The people in the group chat (Sarah and others) have the most complete knowledge",
    ],
    reasoning="Tests ability to track different levels of knowledge across multiple agents.",
    difficulty="medium",
)

BROKEN_VASE = ToMScenario(
    name="broken_vase_knowledge",
    tom_type=ToMType.KNOWLEDGE_ATTRIBUTION,
    setup="A vase is broken in the living room at 2 PM.",
    events=[
        "John was in the living room when it broke and saw it happen.",
        "Mary came home at 3 PM and saw the broken vase.",
        "Peter was at work all day and won't be home until 6 PM.",
        "At 4 PM, Mary calls Peter to tell him about the broken vase but doesn't say how it broke.",
    ],
    test_questions=[
        "At 2:30 PM, who knows the vase is broken?",
        "At 4:30 PM, who knows the vase is broken?",
        "Who knows how the vase broke?",
        "Does Peter know who broke the vase?",
    ],
    correct_answers=[
        "Only John knows at 2:30 PM",
        "John, Mary, and Peter all know by 4:30 PM",
        "Only John knows how it broke (he saw it)",
        "No, Peter only knows it's broken, not how or who",
    ],
    reasoning="Tests temporal tracking of knowledge acquisition and distinguishing between different types/levels of knowledge.",
    difficulty="medium",
)

# ============================================================================
# Perspective Taking Scenarios
# ============================================================================

MOVIE_OPINIONS = ToMScenario(
    name="movie_opinions_perspective",
    tom_type=ToMType.PERSPECTIVE_TAKING,
    setup="Alice loves action movies but dislikes romantic comedies. Bob loves romantic comedies but dislikes action movies. Carol hasn't expressed any preferences.",
    events=[
        "The group is trying to pick a movie to watch tonight.",
        "They're choosing between an action movie and a romantic comedy.",
    ],
    test_questions=[
        "Which movie would Alice prefer?",
        "Which movie would Bob prefer?",
        "If Alice suggests the action movie, how might Bob feel about that?",
        "What would be a fair way to decide?",
    ],
    correct_answers=[
        "Alice would prefer the action movie",
        "Bob would prefer the romantic comedy",
        "Bob might feel disappointed or unheard since it's not his preference",
        "They could take turns choosing, find a compromise genre, or include Carol's preference",
    ],
    reasoning="Tests ability to understand different preferences and predict emotional reactions from different perspectives.",
    difficulty="easy",
)

# ============================================================================
# Belief Updating Scenarios
# ============================================================================

WEATHER_UPDATE = ToMScenario(
    name="weather_belief_updating",
    tom_type=ToMType.BELIEF_UPDATING,
    setup="In the morning, Jake checks his weather app and sees it will be sunny all day. He decides to walk to work without an umbrella.",
    events=[
        "At noon, Jake's colleague mentions that the forecast changed and now predicts rain in the afternoon.",
        "Jake checks his app and confirms the updated forecast.",
    ],
    test_questions=[
        "What did Jake believe about the weather in the morning?",
        "What does Jake believe about the weather after talking to his colleague?",
        "Should Jake's plans change based on his updated belief?",
        "Why did Jake's belief change?",
    ],
    correct_answers=[
        "Jake believed it would be sunny all day",
        "Jake now believes it will rain in the afternoon",
        "Yes, Jake should probably get an umbrella or arrange alternate transportation",
        "Jake's belief changed because he received new, more current information",
    ],
    reasoning="Tests understanding of how beliefs change with new evidence and how this should affect behavior.",
    difficulty="easy",
)

# ============================================================================
# Second-Order Belief Scenarios
# ============================================================================

GIFT_SURPRISE = ToMScenario(
    name="gift_second_order_belief",
    tom_type=ToMType.SECOND_ORDER_BELIEF,
    setup="Lisa is planning to give her friend Mark a book for his birthday. She thinks Mark doesn't know about the gift.",
    events=[
        "Mark accidentally saw Lisa buying the book last week, but Lisa doesn't know that Mark saw her.",
        "Mark decides to act surprised when he receives the gift so Lisa feels good about the surprise.",
    ],
    test_questions=[
        "Does Lisa know Mark saw her buy the book?",
        "Does Mark know that Lisa thinks he doesn't know about the gift?",
        "What does Lisa believe about Mark's knowledge of the gift?",
        "What does Mark believe about Lisa's belief about his knowledge?",
    ],
    correct_answers=[
        "No, Lisa doesn't know Mark saw her",
        "Yes, Mark knows Lisa thinks he doesn't know",
        "Lisa believes Mark doesn't know about the gift",
        "Mark believes Lisa thinks he doesn't know about the gift",
    ],
    reasoning="Tests second-order theory of mind: beliefs about beliefs. Requires tracking nested mental states.",
    difficulty="hard",
)

# ============================================================================
# Epistemic State Scenarios
# ============================================================================

COIN_FLIP = ToMScenario(
    name="coin_flip_epistemic_state",
    tom_type=ToMType.EPISTEMIC_STATE,
    setup="A coin is flipped but lands under a table where no one can see it.",
    events=[
        "Alice says 'I believe it's heads' without looking.",
        "Bob crawls under the table, looks at the coin, and says 'It's heads.'",
        "Carol calculates the probability and says 'There's a 50% chance it's heads.'",
    ],
    test_questions=[
        "Who knows whether the coin is heads or tails?",
        "What is the difference between Alice's and Bob's mental states regarding the coin?",
        "Is Carol's statement about knowledge or about probability?",
        "Can Alice be certain about her belief?",
    ],
    correct_answers=[
        "Only Bob knows (he observed it)",
        "Alice has a belief (guess) while Bob has knowledge (observed truth)",
        "Carol's statement is about probability, not knowledge",
        "No, Alice cannot be certain; she's guessing without evidence",
    ],
    reasoning="Tests ability to distinguish between knowing, believing, and probabilistic reasoning.",
    difficulty="medium",
)

# ============================================================================
# Pragmatic Reasoning Scenarios
# ============================================================================

DOOR_LOCKED = ToMScenario(
    name="locked_door_pragmatics",
    tom_type=ToMType.PRAGMATIC_REASONING,
    setup="Sarah arrives at her friend's house and knocks on the door. Her friend calls from inside: 'The door is unlocked!'",
    events=["Sarah hears this message clearly."],
    test_questions=[
        "What is the friend literally saying?",
        "What is the friend implying Sarah should do?",
        "Why didn't the friend just say 'Come in'?",
        "What knowledge does the friend assume Sarah has?",
    ],
    correct_answers=[
        "The friend is stating that the door is not locked",
        "The friend is implying Sarah should open the door and come in",
        "The friend might be unable to come to the door, or 'door is unlocked' more directly addresses the potential barrier",
        "The friend assumes Sarah knows that an unlocked door means she can enter",
    ],
    reasoning="Tests understanding of pragmatic implication and shared knowledge assumptions in communication.",
    difficulty="medium",
)


# ============================================================================
# Scenario Registry
# ============================================================================

ALL_SCENARIOS = [
    SALLY_ANNE,
    CHOCOLATE_BAR,
    SURPRISE_PARTY,
    BROKEN_VASE,
    MOVIE_OPINIONS,
    WEATHER_UPDATE,
    GIFT_SURPRISE,
    COIN_FLIP,
    DOOR_LOCKED,
]

SCENARIOS_BY_TYPE = {
    ToMType.FALSE_BELIEF: [SALLY_ANNE, CHOCOLATE_BAR],
    ToMType.KNOWLEDGE_ATTRIBUTION: [SURPRISE_PARTY, BROKEN_VASE],
    ToMType.PERSPECTIVE_TAKING: [MOVIE_OPINIONS],
    ToMType.BELIEF_UPDATING: [WEATHER_UPDATE],
    ToMType.SECOND_ORDER_BELIEF: [GIFT_SURPRISE],
    ToMType.EPISTEMIC_STATE: [COIN_FLIP],
    ToMType.PRAGMATIC_REASONING: [DOOR_LOCKED],
}

SCENARIOS_BY_DIFFICULTY = {
    "easy": [s for s in ALL_SCENARIOS if s.difficulty == "easy"],
    "medium": [s for s in ALL_SCENARIOS if s.difficulty == "medium"],
    "hard": [s for s in ALL_SCENARIOS if s.difficulty == "hard"],
}

SCENARIOS_BY_NAME = {s.name: s for s in ALL_SCENARIOS}


def get_scenario(name: str) -> Optional[ToMScenario]:
    """Get a scenario by name"""
    return SCENARIOS_BY_NAME.get(name)


def get_scenarios_by_type(tom_type: ToMType) -> List[ToMScenario]:
    """Get all scenarios of a specific ToM type"""
    return SCENARIOS_BY_TYPE.get(tom_type, [])


def get_scenarios_by_difficulty(difficulty: str) -> List[ToMScenario]:
    """Get all scenarios of a specific difficulty"""
    return SCENARIOS_BY_DIFFICULTY.get(difficulty, [])
