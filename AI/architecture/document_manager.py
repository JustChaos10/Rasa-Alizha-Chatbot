"""
Document Manager - Handles document ingestion into the knowledge base

This module manages the workflow of:
1. Moving uploaded files to the documents directory
2. Triggering vector database rebuild via the knowledgebase MCP server
3. Providing status updates on indexing progress
"""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentManager:
    """
    Manages document ingestion and indexing for the knowledge base.
    """
    
    def __init__(
        self,
        uploads_dir: Optional[Path] = None,
        documents_dir: Optional[Path] = None,
        vectordb_path: Optional[Path] = None,
    ):
        if uploads_dir is None:
            uploads_raw = (os.getenv("UPLOADS_DIR") or os.getenv("UPLOAD_FOLDER") or "uploads").strip()
            uploads_dir = Path(uploads_raw)

        if documents_dir is None:
            documents_raw = (os.getenv("DOCUMENTS_DIR") or "documents").strip()
            documents_dir = Path(documents_raw)

        if vectordb_path is None:
            vectordb_raw = (os.getenv("VECTORDB_PATH") or "").strip()
            vectordb_path = Path(vectordb_raw) if vectordb_raw else None

        self.uploads_dir = uploads_dir
        self.documents_dir = documents_dir
        self.vectordb_path = vectordb_path
        
        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        # Vector DB/index state must live ONLY in the Vector tier. Do not create it here.
        
        # Supported file extensions
        self.supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.pptx', '.csv'}
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the documents directory."""
        documents = []
        
        if not self.documents_dir.exists():
            return documents
        
        for file_path in self.documents_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                documents.append({
                    'name': file_path.name,
                    'path': str(file_path.relative_to(self.documents_dir)),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'extension': file_path.suffix
                })
        
        return sorted(documents, key=lambda x: x['modified'], reverse=True)
    
    def list_uploads(self) -> List[Dict[str, Any]]:
        """List all files in the uploads directory."""
        uploads = []
        
        if not self.uploads_dir.exists():
            return uploads
        
        for file_path in self.uploads_dir.iterdir():
            if file_path.is_file():
                uploads.append({
                    'name': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size,
                    'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    'extension': file_path.suffix,
                    'is_supported': file_path.suffix.lower() in self.supported_extensions
                })
        
        return sorted(uploads, key=lambda x: x['modified'], reverse=True)
    
    def move_to_documents(self, filename: str, source_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Move a file from uploads (or specified directory) to documents directory.
        
        Args:
            filename: Name of the file to move
            source_dir: Source directory (defaults to uploads_dir)
            
        Returns:
            Dict with status and message
        """
        source_dir = source_dir or self.uploads_dir
        source_path = source_dir / filename
        
        if not source_path.exists():
            return {
                'success': False,
                'message': f'File not found: {filename}'
            }
        
        if source_path.suffix.lower() not in self.supported_extensions:
            return {
                'success': False,
                'message': f'Unsupported file type: {source_path.suffix}'
            }
        
        # Destination path
        dest_path = self.documents_dir / filename
        
        # If file already exists in documents, add timestamp
        if dest_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stem = dest_path.stem
            suffix = dest_path.suffix
            dest_path = self.documents_dir / f"{stem}_{timestamp}{suffix}"
        
        try:
            # Move file
            shutil.move(str(source_path), str(dest_path))
            logger.info(f"📄 Moved {filename} to documents directory")
            
            return {
                'success': True,
                'message': f'File moved successfully: {dest_path.name}',
                'path': str(dest_path.relative_to(self.documents_dir))
            }
        except Exception as e:
            logger.error(f"Failed to move file {filename}: {e}")
            return {
                'success': False,
                'message': f'Failed to move file: {str(e)}'
            }
    
    def move_all_uploads(self) -> Dict[str, Any]:
        """
        Move all supported files from uploads to documents directory.
        
        Returns:
            Dict with counts and list of moved files
        """
        uploads = self.list_uploads()
        supported = [u for u in uploads if u['is_supported']]
        
        moved = []
        failed = []
        
        for upload in supported:
            result = self.move_to_documents(upload['name'])
            if result['success']:
                moved.append(upload['name'])
            else:
                failed.append({'name': upload['name'], 'error': result['message']})
        
        return {
            'success': len(failed) == 0,
            'moved_count': len(moved),
            'failed_count': len(failed),
            'moved_files': moved,
            'failed_files': failed
        }
    
    async def trigger_indexing(self, mcp_host=None) -> Dict[str, Any]:
        """
        Trigger the knowledgebase MCP server to rebuild the vector database.
        
        Args:
            mcp_host: Optional MCPHost instance. If not provided, will import and get singleton.
            
        Returns:
            Dict with indexing status
        """
        try:
            # Import here to avoid circular imports
            if mcp_host is None:
                from architecture.mcp_host import get_mcp_host
                mcp_host = get_mcp_host()
            
            # Check if knowledgebase server is connected
            if 'knowledgebase' not in mcp_host.list_servers():
                # Try to connect
                connected = await mcp_host.connect('knowledgebase')
                if not connected:
                    return {
                        'success': False,
                        'message': 'Failed to connect to knowledgebase server'
                    }
            
            # The knowledgebase server automatically indexes documents on startup
            # We can trigger a health check to verify it's working
            result = await mcp_host.call_tool('knowledgebase.knowledgebase_health', {})
            
            if 'error' in result:
                return {
                    'success': False,
                    'message': f'Indexing check failed: {result.get("error")}'
                }
            
            # Parse the health check result
            import json
            health = json.loads(result) if isinstance(result, str) else result
            
            return {
                'success': True,
                'message': 'Vector database is ready',
                'status': health.get('status'),
                'vectordb_available': health.get('vectordb_available'),
                'docs_path': health.get('docs_path')
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger indexing: {e}")
            return {
                'success': False,
                'message': f'Indexing error: {str(e)}'
            }
    
    async def process_and_index(self, mcp_host=None) -> Dict[str, Any]:
        """
        Complete workflow: move uploads to documents and trigger indexing.
        
        Returns:
            Dict with complete status
        """
        # Step 1: Move files
        move_result = self.move_all_uploads()
        
        # Step 2: Trigger indexing
        index_result = await self.trigger_indexing(mcp_host)
        
        return {
            'success': move_result['success'] and index_result['success'],
            'files_moved': move_result['moved_count'],
            'files_failed': move_result['failed_count'],
            'moved_files': move_result['moved_files'],
            'failed_files': move_result['failed_files'],
            'indexing_status': index_result.get('status'),
            'vectordb_ready': index_result.get('vectordb_available', False),
            'message': f"Moved {move_result['moved_count']} files. {index_result.get('message', '')}"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of documents and uploads."""
        documents = self.list_documents()
        uploads = self.list_uploads()
        supported_uploads = [u for u in uploads if u['is_supported']]

        vectordb_exists = False
        try:
            if self.vectordb_path is not None:
                vectordb_exists = (self.vectordb_path / 'index.faiss').exists()
        except Exception:
            vectordb_exists = False
        
        return {
            'documents_count': len(documents),
            'uploads_count': len(uploads),
            'supported_uploads_count': len(supported_uploads),
            'vectordb_exists': vectordb_exists,
            'documents': documents,
            'uploads': supported_uploads
        }


# Global instance
_document_manager: Optional[DocumentManager] = None


def get_document_manager() -> DocumentManager:
    """Get or create the global DocumentManager instance."""
    global _document_manager
    if _document_manager is None:
        _document_manager = DocumentManager()
    return _document_manager
