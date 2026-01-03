"""
File Tool Plugin - Process uploaded files (images and documents).

Handles image analysis (vision) and document processing (PDF, DOCX, TXT).
Supports:
- Language detection (Arabic/English)
- Sticky context for follow-up questions
- Concise 30-50 word summaries
"""

import logging
from pathlib import Path
from typing import Any, Dict

from architecture.base_tool import BaseTool, ToolSchema

logger = logging.getLogger(__name__)

UPLOAD_FOLDER = Path("uploads")


class FileTool(BaseTool):
    """
    Tool for processing uploaded files with sticky context.
    
    Supports:
    - Image analysis using vision models
    - Document processing (PDF, Word, Text)
    - Follow-up questions about last processed file
    - Language-aware responses (Arabic/English)
    """
    
    def __init__(self):
        self._file_service = None
    
    def _get_file_service(self):
        """Lazy load file service."""
        if self._file_service is None:
            from shared_utils import get_service_manager
            self._file_service = get_service_manager().get_file_service()
        return self._file_service
    
    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file",
            description="""Process uploaded files (images/documents). Supports Arabic and English with automatic language detection.
            
Actions:
- 'analyze_image': Analyze an uploaded image (30-50 word summary)
- 'process_document': Summarize a PDF/DOCX/TXT (30-50 words in detected language)
- 'followup': Ask follow-up questions about the last processed file
- 'translate_summary': Get summary in different language (ar/en)
- 'list_files': Show recent uploads
- 'instructions': How to upload files""",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": ["analyze_image", "process_document", "followup", "translate_summary", "list_files", "instructions"]
                    },
                    "file_path": {
                        "type": "string",
                        "description": "Optional specific file path to process"
                    },
                    "question": {
                        "type": "string",
                        "description": "Question about the file content (for followup action)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Target language: 'ar' for Arabic, 'en' for English",
                        "enum": ["ar", "en"]
                    }
                },
                "required": ["action"]
            },
            examples=[
                "Analyze the image I uploaded",
                "Summarize the PDF",
                "What does this document say?",
                "Summarize in English",
                "لخص بالعربي",
                "Tell me more about the document",
                "What's the main point?",
                "Translate the summary to Arabic",
                "Describe the uploaded picture"
            ],
            input_examples=[
                {"action": "analyze_image"},
                {"action": "process_document"},
                {"action": "followup", "question": "What is the main topic?"},
                {"action": "translate_summary", "language": "en"},
                {"action": "list_files"}
            ],
            defer_loading=True,
            always_loaded=False
        )
    
    async def execute(
        self,
        action: str,
        file_path: str = "",
        question: str = "",
        language: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process uploaded files with language support and sticky context.
        
        Args:
            action: The action to perform
            file_path: Optional specific file path
            question: Question about the content (for followup)
            language: Target language ('ar' or 'en')
            
        Returns:
            Processing result with sticky_context for follow-ups
        """
        try:
            if action == "instructions":
                return self._get_upload_instructions()
            
            if action == "list_files":
                return await self._list_recent_files()
            
            if action == "analyze_image":
                return await self._analyze_image(file_path, question, language)
            
            if action == "process_document":
                return await self._process_document(file_path, question, language)
            
            if action == "followup":
                return await self._handle_followup(question, language)
            
            if action == "translate_summary":
                return await self._translate_summary(language)
            
            return {
                "success": False,
                "data": f"Unknown action: {action}. Use 'analyze_image', 'process_document', 'followup', 'translate_summary', 'list_files', or 'instructions'."
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "data": f"❌ Error processing file: {str(e)}"
            }
    
    def _get_upload_instructions(self) -> Dict[str, Any]:
        """Return file upload instructions."""
        return {
            "success": True,
            "data": """📁 **File Upload Instructions**

I can analyze images and documents in **Arabic** and **English**!

**Supported Files:**
• 🖼️ Images: JPG, PNG, GIF, WebP
• 📄 Documents: PDF, DOCX, TXT

**How to Use:**
1. Click the 📎 attachment button
2. Select your file
3. I'll automatically detect the language and summarize

**Follow-up Commands:**
• "Tell me more about this"
• "Summarize in English" / "لخص بالعربي"
• "What's the main point?"

**Language Support:**
• Arabic documents → Arabic summaries (العربية)
• English documents → English summaries
• You can ask for translation anytime!
"""
        }
    
    async def _list_recent_files(self) -> Dict[str, Any]:
        """List recently uploaded files."""
        try:
            file_service = self._get_file_service()
            
            image_files = file_service.find_recent_files('image')
            doc_files = file_service.find_recent_files('document')
            
            lines = ["📂 **Recent Uploads**\n"]
            
            if image_files:
                lines.append("🖼️ **Images:**")
                for f in image_files[:5]:
                    lines.append(f"  • {Path(f).name}")
            
            if doc_files:
                lines.append("\n📄 **Documents:**")
                for f in doc_files[:5]:
                    lines.append(f"  • {Path(f).name}")
            
            if not image_files and not doc_files:
                lines.append("No recent files found. Upload a file to get started!")
            
            # Show current context if available
            context = file_service.get_last_file_context()
            if context.get("file_path"):
                lang = "Arabic" if context.get("language") == "ar" else "English"
                lines.append(f"\n🔗 **Active context:** {Path(context['file_path']).name} ({lang})")
                lines.append("Ask follow-up questions about this file!")
            
            return {
                "success": True,
                "data": "\n".join(lines)
            }
        except Exception as e:
            return {
                "success": False,
                "data": f"❌ Could not list files: {str(e)}"
            }
    
    async def _analyze_image(self, file_path: str, question: str, language: str) -> Dict[str, Any]:
        """Analyze an uploaded image."""
        file_service = self._get_file_service()
        
        # Find image file
        if file_path:
            image_file = file_path
        else:
            image_files = file_service.find_recent_files('image')
            if not image_files:
                return {
                    "success": False,
                    "data": "🖼️ No recent images found. Please upload an image first."
                }
            image_file = image_files[0]
        
        # Process the image with language preference
        result = file_service.process_image(image_file, target_language=language or None)
        
        if result:
            from shared_utils import MessageFormatter
            clean_result = MessageFormatter.clean_markdown_text(result)
            
            # Get detected language from context
            context = file_service.get_last_file_context()
            detected_lang = context.get("language", "en")
            
            # Return with sticky context
            return {
                "success": True,
                "data": f"🖼️ **Image Analysis:**\n\n{clean_result}",
                "sticky_context": {
                    "tool": "file",
                    "file_path": str(image_file),
                    "file_type": "image",
                    "language": detected_lang
                }
            }
        else:
            return {
                "success": False,
                "data": "❌ I couldn't analyze the image. Please ensure it's a valid image file."
            }
    
    async def _process_document(self, file_path: str, question: str, language: str) -> Dict[str, Any]:
        """Process an uploaded document with language detection."""
        file_service = self._get_file_service()
        
        # Find document file
        if file_path:
            doc_file = file_path
        else:
            doc_files = file_service.find_recent_files('document')
            if not doc_files:
                return {
                    "success": False,
                    "data": "📄 No recent documents found. Please upload a document first."
                }
            doc_file = doc_files[0]
        
        # Process the document with language preference
        result = file_service.process_document(doc_file, target_language=language or None)
        
        if result:
            from shared_utils import MessageFormatter
            clean_result = MessageFormatter.clean_markdown_text(result)
            
            # Get detected language
            context = file_service.get_last_file_context()
            detected_lang = context.get("language", "en")
            lang_label = "العربية" if detected_lang == "ar" else "English"
            
            return {
                "success": True,
                "data": f"📄 **Summary** ({lang_label}):\n\n{clean_result}\n\n💡 *Ask follow-up questions or request translation!*",
                "sticky_context": {
                    "tool": "file",
                    "file_path": str(doc_file),
                    "file_type": "document",
                    "language": detected_lang
                }
            }
        else:
            return {
                "success": False,
                "data": "❌ I couldn't process the document. It may be empty, corrupted, or in an unsupported format."
            }
    
    async def _handle_followup(self, question: str, language: str) -> Dict[str, Any]:
        """Handle follow-up questions about the last processed file."""
        file_service = self._get_file_service()
        context = file_service.get_last_file_context()
        
        if not context.get("file_path"):
            return {
                "success": False,
                "data": "📄 No file context available. Please upload a file first!"
            }
        
        if not question:
            return {
                "success": False,
                "data": "❓ What would you like to know about the file? Ask me a question!"
            }
        
        result = file_service.answer_followup(question, target_language=language or None)
        
        if result:
            from shared_utils import MessageFormatter
            clean_result = MessageFormatter.clean_markdown_text(result)
            
            file_name = Path(context["file_path"]).name
            return {
                "success": True,
                "data": f"📄 **Re: {file_name}**\n\n{clean_result}",
                "sticky_context": {
                    "tool": "file",
                    "file_path": context["file_path"],
                    "file_type": context.get("file_type", "document"),
                    "language": language or context.get("language", "en")
                }
            }
        else:
            return {
                "success": False,
                "data": "❌ I couldn't answer that question. Please try rephrasing."
            }
    
    async def _translate_summary(self, language: str) -> Dict[str, Any]:
        """Translate/re-summarize the last file in a different language."""
        file_service = self._get_file_service()
        context = file_service.get_last_file_context()
        
        if not context.get("file_path"):
            return {
                "success": False,
                "data": "📄 No file context available. Please upload a file first!"
            }
        
        target_lang = language or ("en" if context.get("language") == "ar" else "ar")
        lang_label = "العربية" if target_lang == "ar" else "English"
        
        # Re-process with new language
        file_path = Path(context["file_path"])
        if file_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
            result = file_service.process_image(file_path, target_language=target_lang)
        else:
            result = file_service.process_document(file_path, target_language=target_lang)
        
        if result:
            from shared_utils import MessageFormatter
            clean_result = MessageFormatter.clean_markdown_text(result)
            
            return {
                "success": True,
                "data": f"🌐 **Summary in {lang_label}:**\n\n{clean_result}",
                "sticky_context": {
                    "tool": "file",
                    "file_path": str(file_path),
                    "language": target_lang
                }
            }
        else:
            return {
                "success": False,
                "data": f"❌ Couldn't translate to {lang_label}. Please try again."
            }
