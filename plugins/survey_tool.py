"""
Survey Tool Plugin - Conversational Survey Collection.

Generates dynamic surveys based on conversation history and collects
responses one question at a time using a conversational chain approach.
All data is stored in Redis for session persistence.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)


class SurveyTool(BaseTool):
    """
    Tool for generating and conducting conversational surveys.
    
    Flow:
    1. User triggers survey -> LLM generates questions based on conversation history
    2. Bot asks first question
    3. User responds -> Bot stores answer and asks next question
    4. Repeat until all questions answered
    5. Store complete survey in Redis and show summary
    
    Uses sticky context to maintain survey state across turns.
    """
    
    # Survey states
    STATE_IDLE = "idle"
    STATE_COLLECTING = "collecting"
    STATE_COMPLETE = "complete"
    
    def __init__(self):
        self._llm_service = None
        self._model = "llama-3.3-70b-versatile"
        self._redis_client = None
    
    def _get_llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from shared_utils import get_service_manager
            self._llm_service = get_service_manager().get_llm_service()
        return self._llm_service
    
    async def _get_redis(self):
        """Get Redis client for survey storage."""
        if self._redis_client is None:
            import redis.asyncio as redis
            from config.config import ConfigManager
            config = ConfigManager()
            self._redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
                password=config.redis_password,
                decode_responses=True
            )
        return self._redis_client
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="survey",
            description="Generate and conduct conversational surveys. Creates questions based on conversation history and collects responses one at a time. Use 'start' to begin a new survey, 'collect' to process user responses during an active survey.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action: 'start' to begin survey, 'collect' to process response, 'status' to check progress, 'results' to view completed surveys",
                        "enum": ["start", "collect", "status", "results"]
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic for the survey (used with 'start' action)"
                    },
                    "num_questions": {
                        "type": "integer",
                        "description": "Number of questions (1-10, default 3)",
                        "default": 3
                    },
                    "user_response": {
                        "type": "string",
                        "description": "User's response to current question (used with 'collect' action)"
                    },
                    "conversation_history": {
                        "type": "string",
                        "description": "Previous conversation context for generating relevant questions"
                    }
                },
                "required": ["action"]
            },
            examples=[
                "Start a survey",
                "Generate a survey about our conversation",
                "Create feedback questions",
                "Begin questionnaire",
                "Survey me",
                "Collect my feedback",
                "Ask me some questions"
            ],
            input_examples=[
                {"action": "start", "topic": "conversation feedback"},
                {"action": "start", "num_questions": 5},
                {"action": "collect", "user_response": "I really liked the feature"},
                {"action": "status"},
                {"action": "results"}
            ],
            defer_loading=True,
            always_loaded=False
        )
    
    async def execute(
        self,
        action: str,
        topic: str = "",
        num_questions: int = 3,
        user_response: str = "",
        conversation_history: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute survey actions.
        
        Args:
            action: 'start', 'collect', 'status', or 'results'
            topic: Survey topic (for 'start')
            num_questions: Number of questions (for 'start')
            user_response: User's answer (for 'collect')
            conversation_history: Previous conversation context
            
        Returns:
            Dict with response and optional sticky_context for chain
        """
        # Get sender_id from kwargs (passed by router)
        sender_id = kwargs.get("sender_id", kwargs.get("user_id", kwargs.get("_sender_id", "anonymous")))
        
        # Get conversation history (might be under different keys)
        conv_history = conversation_history or kwargs.get("_conversation_history", "")
        
        try:
            if action == "start":
                return await self._start_survey(
                    sender_id, 
                    topic, 
                    num_questions, 
                    conv_history
                )
            
            elif action == "collect":
                return await self._collect_response(
                    sender_id,
                    user_response,
                    kwargs.get("current_step"),
                    kwargs.get("survey_data", {})
                )
            
            elif action == "status":
                return await self._get_status(sender_id)
            
            elif action == "results":
                return await self._get_results(sender_id)
            
            else:
                return {
                    "success": False,
                    "data": "Unknown action. Use 'start', 'collect', 'status', or 'results'."
                }
                
        except Exception as e:
            logger.error(f"Survey error: {e}", exc_info=True)
            return {
                "success": False,
                "data": f"❌ Survey error: {str(e)}"
            }
    
    async def _start_survey(
        self,
        sender_id: str,
        topic: str,
        num_questions: int,
        conversation_history: str
    ) -> Dict[str, Any]:
        """
        Start a new survey by generating questions from conversation history.
        """
        logger.info(f"📋 Starting survey for {sender_id}, topic: {topic}")
        
        # Clamp question count
        num_questions = max(1, min(10, num_questions))
        
        # Generate questions based on conversation history
        questions = await self._generate_questions(
            topic or "conversation feedback",
            num_questions,
            conversation_history
        )
        
        if not questions:
            return {
                "success": False,
                "data": "I couldn't generate survey questions. Please try again."
            }
        
        # Create survey session data
        survey_data = {
            "survey_id": f"survey_{sender_id}_{int(time.time())}",
            "topic": topic or "conversation feedback",
            "questions": questions,
            "responses": {},
            "current_step": 0,
            "total_steps": len(questions),
            "state": self.STATE_COLLECTING,
            "started_at": time.time(),
            "sender_id": sender_id
        }
        
        # Store in Redis
        await self._save_survey_session(sender_id, survey_data)
        
        # Ask first question
        first_question = questions[0]
        
        return {
            "success": True,
            "data": f"📋 **Survey Started: {survey_data['topic']}**\n\n"
                   f"I have {len(questions)} questions for you. Let's begin!\n\n"
                   f"**Question 1/{len(questions)}:** {first_question['text']}",
            "sticky_context": {
                "tool_name": "survey",
                "state": {
                    "action": "collect",
                    "current_step": 0,
                    "survey_data": survey_data
                }
            }
        }
    
    async def _collect_response(
        self,
        sender_id: str,
        user_response: str,
        current_step: Optional[int],
        survey_data: Dict
    ) -> Dict[str, Any]:
        """
        Collect user's response and move to next question.
        """
        # Load survey session if not provided
        if not survey_data or not survey_data.get("questions"):
            survey_data = await self._load_survey_session(sender_id)
            if not survey_data:
                return {
                    "success": False,
                    "data": "No active survey found. Say 'start survey' to begin one."
                }
        
        current_step = current_step if current_step is not None else survey_data.get("current_step", 0)
        questions = survey_data.get("questions", [])
        
        if current_step >= len(questions):
            return await self._complete_survey(sender_id, survey_data)
        
        # Store the response
        current_question = questions[current_step]
        survey_data["responses"][current_question["id"]] = {
            "question": current_question["text"],
            "answer": user_response,
            "answered_at": time.time()
        }
        
        # Move to next step
        next_step = current_step + 1
        survey_data["current_step"] = next_step
        
        # Check if survey is complete
        if next_step >= len(questions):
            return await self._complete_survey(sender_id, survey_data)
        
        # Save updated session
        await self._save_survey_session(sender_id, survey_data)
        
        # Ask next question
        next_question = questions[next_step]
        
        return {
            "success": True,
            "data": f"✅ Got it!\n\n**Question {next_step + 1}/{len(questions)}:** {next_question['text']}",
            "sticky_context": {
                "tool_name": "survey",
                "state": {
                    "action": "collect",
                    "current_step": next_step,
                    "survey_data": survey_data
                }
            }
        }
    
    async def _complete_survey(
        self,
        sender_id: str,
        survey_data: Dict
    ) -> Dict[str, Any]:
        """
        Complete the survey and store results.
        """
        survey_data["state"] = self.STATE_COMPLETE
        survey_data["completed_at"] = time.time()
        
        # Store completed survey in Redis (with different key for history)
        await self._save_completed_survey(sender_id, survey_data)
        
        # Clear active session
        await self._clear_survey_session(sender_id)
        
        # Build summary
        summary_lines = [
            f"🎉 **Survey Complete: {survey_data.get('topic', 'Feedback')}**\n",
            "Thank you for your responses! Here's a summary:\n"
        ]
        
        for i, (q_id, response) in enumerate(survey_data.get("responses", {}).items(), 1):
            summary_lines.append(f"**Q{i}:** {response['question']}")
            summary_lines.append(f"**A:** {response['answer']}\n")
        
        summary_lines.append("\n✅ Your feedback has been saved!")
        
        return {
            "success": True,
            "data": "\n".join(summary_lines),
            "sticky_context": None  # Clear sticky context
        }
    
    async def _get_status(self, sender_id: str) -> Dict[str, Any]:
        """Get current survey status."""
        survey_data = await self._load_survey_session(sender_id)
        
        if not survey_data:
            return {
                "success": True,
                "data": "No active survey. Say 'start survey' to begin one."
            }
        
        current = survey_data.get("current_step", 0)
        total = survey_data.get("total_steps", 0)
        answered = len(survey_data.get("responses", {}))
        
        return {
            "success": True,
            "data": f"📊 **Survey Status**\n\n"
                   f"Topic: {survey_data.get('topic', 'N/A')}\n"
                   f"Progress: {answered}/{total} questions answered\n"
                   f"Current question: {current + 1}"
        }
    
    async def _get_results(self, sender_id: str) -> Dict[str, Any]:
        """Get completed survey results."""
        redis_client = await self._get_redis()
        
        # Get all completed surveys for this user
        pattern = f"survey_completed:{sender_id}:*"
        keys = []
        async for key in redis_client.scan_iter(match=pattern):
            keys.append(key)
        
        if not keys:
            return {
                "success": True,
                "data": "No completed surveys found."
            }
        
        results = []
        for key in sorted(keys, reverse=True)[:5]:  # Last 5 surveys
            data = await redis_client.get(key)
            if data:
                survey = json.loads(data)
                results.append(survey)
        
        # Format results
        lines = ["📋 **Your Completed Surveys**\n"]
        
        for survey in results:
            topic = survey.get("topic", "Survey")
            completed_at = survey.get("completed_at", 0)
            response_count = len(survey.get("responses", {}))
            
            from datetime import datetime
            date_str = datetime.fromtimestamp(completed_at).strftime("%Y-%m-%d %H:%M") if completed_at else "Unknown"
            
            lines.append(f"• **{topic}** - {response_count} responses ({date_str})")
        
        return {
            "success": True,
            "data": "\n".join(lines)
        }
    
    async def _generate_questions(
        self,
        topic: str,
        num_questions: int,
        conversation_history: str
    ) -> List[Dict[str, Any]]:
        """
        Generate survey questions using LLM based on conversation history.
        """
        logger.info(f"📋 Generating questions - topic: {topic}, history length: {len(conversation_history) if conversation_history else 0}")
        
        # Build context-aware prompt
        context_section = ""
        if conversation_history and conversation_history.strip():
            # Truncate but keep enough context
            history_text = conversation_history[:3000]
            context_section = f"""
IMPORTANT: Generate questions based SPECIFICALLY on this conversation history.
DO NOT generate generic feedback questions. Questions MUST reference specific topics discussed.

=== CONVERSATION HISTORY ===
{history_text}
=== END CONVERSATION ===

Analyze what was discussed above and generate questions about those SPECIFIC topics.
For example:
- If they discussed weather, ask about the weather information provided
- If they discussed leave requests, ask about the leave process
- If they used specific tools, ask about their experience with those features
"""
            logger.debug(f"Using conversation history for question generation")
        else:
            logger.warning(f"No conversation history provided for survey question generation")
        
        prompt = f"""Generate {num_questions} survey questions about: {topic}
{context_section}
Requirements:
1. Questions MUST be specific to what was discussed in the conversation
2. DO NOT use generic questions like "How was your experience?"
3. Reference specific topics, tools, or information from the conversation
4. Mix of question types: some open-ended, some rating (1-5 scale)
5. Questions should be concise and clear

Return ONLY a JSON array:
[
    {{"id": "q1", "text": "Question text here", "type": "text"}},
    {{"id": "q2", "text": "Rate your experience", "type": "rating"}},
    ...
]

Types: "text" (open-ended), "rating" (1-5 scale), "yesno" (yes/no)
"""

        messages = [
            {
                "role": "system",
                "content": "You are a survey expert. Generate relevant, unbiased questions based on conversation context. Return only valid JSON array."
            },
            {"role": "user", "content": prompt}
        ]
        
        try:
            llm = self._get_llm_service()
            response = llm.generate_text(
                messages=messages,
                model=self._model,
                max_tokens=600,
                temperature=0.4,
                response_format={"type": "json_object"},
                trace_name="survey-question-generation"
            )
            
            # Parse response
            # Handle if LLM returns {"questions": [...]} or just [...]
            parsed = json.loads(response)
            if isinstance(parsed, list):
                questions = parsed
            elif isinstance(parsed, dict) and "questions" in parsed:
                questions = parsed["questions"]
            else:
                questions = list(parsed.values())[0] if parsed else []
            
            # Validate and fix question structure
            valid_questions = []
            for i, q in enumerate(questions[:num_questions]):
                if isinstance(q, dict) and q.get("text"):
                    valid_questions.append({
                        "id": q.get("id", f"q{i+1}"),
                        "text": q["text"],
                        "type": q.get("type", "text")
                    })
            
            if valid_questions:
                logger.info(f"✅ Generated {len(valid_questions)} survey questions")
                return valid_questions
            
        except Exception as e:
            logger.warning(f"Question generation failed: {e}")
        
        # Fallback questions
        return self._get_fallback_questions(topic, num_questions)
    
    def _get_fallback_questions(self, topic: str, count: int) -> List[Dict[str, Any]]:
        """Return fallback questions if LLM generation fails."""
        fallback = [
            {"id": "q1", "text": f"What did you find most interesting about {topic}?", "type": "text"},
            {"id": "q2", "text": "How would you rate your overall experience? (1-5)", "type": "rating"},
            {"id": "q3", "text": "What could be improved?", "type": "text"},
            {"id": "q4", "text": "Would you recommend this to others? (yes/no)", "type": "yesno"},
            {"id": "q5", "text": "Any additional comments or feedback?", "type": "text"},
        ]
        return fallback[:count]
    
    # ==================== Redis Storage Methods ====================
    
    async def _save_survey_session(self, sender_id: str, data: Dict) -> None:
        """Save active survey session to Redis."""
        redis_client = await self._get_redis()
        key = f"survey_active:{sender_id}"
        await redis_client.set(key, json.dumps(data), ex=3600)  # 1 hour TTL
        logger.debug(f"Saved survey session for {sender_id}")
    
    async def _load_survey_session(self, sender_id: str) -> Optional[Dict]:
        """Load active survey session from Redis."""
        redis_client = await self._get_redis()
        key = f"survey_active:{sender_id}"
        data = await redis_client.get(key)
        if data:
            return json.loads(data)
        return None
    
    async def _clear_survey_session(self, sender_id: str) -> None:
        """Clear active survey session from Redis."""
        redis_client = await self._get_redis()
        key = f"survey_active:{sender_id}"
        await redis_client.delete(key)
        logger.debug(f"Cleared survey session for {sender_id}")
    
    async def _save_completed_survey(self, sender_id: str, data: Dict) -> None:
        """Save completed survey to Redis for history."""
        redis_client = await self._get_redis()
        survey_id = data.get("survey_id", f"survey_{int(time.time())}")
        key = f"survey_completed:{sender_id}:{survey_id}"
        await redis_client.set(key, json.dumps(data), ex=86400 * 30)  # 30 days TTL
        logger.info(f"✅ Saved completed survey: {survey_id}")
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """Format the survey response for display."""
        if not result.get("success", False):
            return result.get("data", "Survey error occurred.")
        return result.get("data", "")
