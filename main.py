import os
from dotenv import load_dotenv
from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Load environment variables
load_dotenv()

# Default configuration for accuracy optimization
default_config = {
    "model": "claude-3-haiku-20240307",
    "chunk_size": 800,  # Smaller chunks for better precision
    "chunk_overlap": 300,  # Higher overlap to prevent information loss
    "retrieval_count": 8,  # More documents retrieved
    "evaluation_enabled": True,
    "temperature": 0.1,
    "search_strategy": "mmr",  # Use MMR for diverse results
    "confidence_threshold": 0.5,  # Lower threshold to get more documents
    "prompt_template": "default",
    "max_context_length": 6000  # Larger context window
}

def authenticate_drive(user_id="default"):
    """Authenticate with Google Drive API."""
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
    token_path = f"tokens/{user_id}_token.json"
    
    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            if creds and not creds.expired:
                print("Found existing token.")
                return creds
            else:
                print("Token corrupted. Reauthenticating...")
        except Exception:
            print("Token corrupted. Reauthenticating...")
    
    print("No valid token. Starting authentication flow...")
    flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
    creds = flow.run_local_server(port=0)
    
    # Save the credentials for the next run
    os.makedirs("tokens", exist_ok=True)
    with open(token_path, 'w') as token:
        token.write(creds.to_json())
    print("New token stored.")
    
    return creds

def list_drive_files(user_id="default"):
    """List all files in Google Drive for the user."""
    print(f"Listing all files in Google Drive for user: {user_id}")
    
    try:
        creds = authenticate_drive(user_id)
        service = build('drive', 'v3', credentials=creds)
        
        # Get all files from all accessible locations
        all_files = []
        
        # 1. Personal Drive files
        try:
            personal_files = service.files().list(
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, parents, size, modifiedTime, webViewLink)"
            ).execute()
            all_files.extend(personal_files.get('files', []))
        except Exception as e:
            print(f"Error accessing personal drive: {e}")
        
        # 2. Shared Drive files
        try:
            shared_drives = service.drives().list(pageSize=100).execute()
            for drive in shared_drives.get('drives', []):
                try:
                    drive_files = service.files().list(
                        pageSize=1000,
                        corpora='drive',
                        driveId=drive['id'],
                        includeItemsFromAllDrives=True,
                        supportsAllDrives=True,
                        fields="nextPageToken, files(id, name, mimeType, parents, size, modifiedTime, webViewLink)"
                    ).execute()
                    all_files.extend(drive_files.get('files', []))
                except Exception as e:
                    print(f"Error accessing shared drive {drive['name']}: {e}")
        except Exception as e:
            print(f"Error listing shared drives: {e}")
        
        if not all_files:
            print("No files found in any accessible Google Drive location")
            return []
        
        # Filter for supported file types
        supported_types = [
            'application/vnd.google-apps.document',
            'application/vnd.google-apps.spreadsheet',
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        ]
        
        supported_files = [f for f in all_files if f.get('mimeType') in supported_types]
        
        # Group files by type
        files_by_type = {}
        for file in supported_files:
            mime_type = file.get('mimeType', 'unknown')
            if mime_type not in files_by_type:
                files_by_type[mime_type] = []
            files_by_type[mime_type].append(file)
        
        # Print summary by file type
        print("\nFile Summary by Type:")
        for mime_type, files in files_by_type.items():
            type_name = mime_type.split('.')[-1].upper()
            print(f"  {type_name}: {len(files)} files")
            for file in files[:3]:  # Show first 3 files of each type
                print(f"    - {file.get('name', 'Unknown')}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more")
        
        # Group files by source (personal vs shared drives)
        personal_files = [f for f in supported_files if not f.get('parents') or 'drive' not in f.get('parents', [''])[0]]
        shared_files = [f for f in supported_files if f not in personal_files]
        
        print("\nFile Summary by Source:")
        print(f"  Personal Drive: {len(personal_files)} files")
        print(f"  Shared Drives: {len(shared_files)} files")
        
        print(f"\nSupported files for loading: {len(supported_files)}")
        
        if supported_files:
            print("Supported files:")
            for file in supported_files[:10]:  # Show first 10 files
                print(f"  - {file.get('name', 'Unknown')} ({file.get('mimeType', 'unknown')})")
            if len(supported_files) > 10:
                print(f"  ... and {len(supported_files) - 10} more")
        
        return supported_files
        
    except HttpError as error:
        print(f"Google Drive API error: {error}")
        return []
    except Exception as e:
        print(f"Error listing files: {e}")
        return []

def test_comprehensive_loading(user_id="default"):
    """
    Test comprehensive loading to ensure all files are captured.
    """
    print(f"Testing comprehensive loading for user: {user_id}")
    creds = authenticate_drive(user_id)
    
    # Test different configurations
    configs = [
        {
            "name": "Standard Loader",
            "config": {
                "folder_id": "root",
                "recursive": True,
                "file_types": ["document", "sheet", "pdf", "presentation"],
                "supports_shared_drives": True,
                "include_items_from_all_drives": True,
                "drive_id": None
            }
        },
        {
            "name": "All Files Loader",
            "config": {
                "folder_id": "root",
                "recursive": True,
                "file_types": ["document", "sheet", "pdf", "presentation"],
                "supports_shared_drives": True,
                "include_items_from_all_drives": True,
                "drive_id": None
            }
        }
    ]
    
    all_docs = []
    
    for config in configs:
        try:
            print(f"\nTesting {config['name']}...")
            loader = GoogleDriveLoader(
                credentials=creds,
                **config['config']
            )
            
            docs = loader.load()
            print(f"Loaded {len(docs)} documents")
            
            # Add to all docs
            for doc in docs:
                if hasattr(doc, 'metadata'):
                    doc.metadata['loader_config'] = config['name']
                all_docs.append(doc)
                
        except Exception as e:
            print(f"Failed: {config['name']} - {e}")
    
    # Remove duplicates
    seen_contents = set()
    unique_docs = []
    for doc in all_docs:
        content_hash = hash(doc.page_content[:200])  # Use first 200 chars as hash
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_docs.append(doc)
    
    print(f"\nComprehensive loading results:")
    print(f"  Total documents found: {len(unique_docs)}")
    print(f"  Duplicates removed: {len(all_docs) - len(unique_docs)}")
    
    return unique_docs

def load_documents(user_id="default"):
    print(f"Loading Google Drive documents for user: {user_id}")
    creds = authenticate_drive(user_id)
    
    # First, let's see what files are available
    print("Checking available files first...")
    available_files = list_drive_files(user_id)
    
    # Enhanced Google Drive loader to collect from ALL accessible locations
    loader = GoogleDriveLoader(
        credentials=creds,
        folder_id="root",  # Use root but enable all drives access
        recursive=True,
        file_types=["document", "sheet", "pdf", "presentation"],
        supports_shared_drives=True,  # Enable shared drives
        include_items_from_all_drives=True,  # Include all drives
        drive_id=None  # None means scan all drives
    )
    
    try:
        print("Scanning ALL accessible Google Drive locations...")
        print("   - Personal Drive")
        print("   - Shared Drives")
        print("   - Shared Folders")
        print("   - All Subfolders")

        docs = loader.load()
        # Detailed logging of what was found
        print(f"Total documents loaded: {len(docs)}")
        if docs:
            # Group documents by type and source
            doc_types = {}
            doc_sources = {}
            total_size = 0
            loaded_files = []
            for doc in docs:
                # Get file type and source from metadata
                file_type = "unknown"
                file_name = "unknown"
                file_size = 0
                file_source = "unknown"
                file_id = "unknown"
                if hasattr(doc, 'metadata'):
                    file_type = doc.metadata.get('file_type', 'unknown')
                    file_name = doc.metadata.get('file_name', 'unknown')
                    file_size = len(doc.page_content) if doc.page_content else 0
                    file_id = doc.metadata.get('file_id', 'unknown')
                    # Determine source
                    if doc.metadata.get('drive_id'):
                        file_source = f"Shared Drive: {doc.metadata.get('drive_id')}"
                    elif doc.metadata.get('parents'):
                        file_source = "Personal Drive"
                    else:
                        file_source = "Unknown Source"
                # Track loaded files
                loaded_files.append({
                    'name': file_name,
                    'type': file_type,
                    'source': file_source,
                    'size': file_size,
                    'id': file_id
                })
                # Group by type
                if file_type not in doc_types:
                    doc_types[file_type] = []
                doc_types[file_type].append(file_name)
                # Group by source
                if file_source not in doc_sources:
                    doc_sources[file_source] = []
                doc_sources[file_source].append(file_name)
                total_size += file_size
            # Print summary by file type
            print("\nDocument Summary by Type:")
            for doc_type, files in doc_types.items():
                print(f"  {doc_type.upper()}: {len(files)} files")
                for file_name in files[:3]:  # Show first 3 files of each type
                    print(f"    - {file_name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
            # Print summary by source
            print("\nDocument Summary by Source:")
            for source, files in doc_sources.items():
                print(f"  {source}: {len(files)} files")
                for file_name in files[:3]:  # Show first 3 files from each source
                    print(f"    - {file_name}")
                if len(files) > 3:
                    print(f"    ... and {len(files) - 3} more")
            print(f"\nTotal content size: {total_size:,} characters")
            # Compare with available files
            if available_files:
                print(f"\nComparison with available files:")
                print(f"  Available supported files: {len(available_files)}")
                print(f"  Actually loaded: {len(loaded_files)}")
                
                if len(loaded_files) < len(available_files):
                    print(f"  {len(available_files) - len(loaded_files)} files were not loaded!")
                    print("  Troubleshooting suggestions:")
                    print("    - Files are too large")
                    print("    - Files are corrupted")
                    print("    - Permission issues")
                    print("    - API rate limiting")
                    
                    # Show which files were not loaded
                    loaded_names = [f['name'] for f in loaded_files]
                    missing_files = [f for f in available_files if f['name'] not in loaded_names]
                    if missing_files:
                        print("  Missing files:")
                        for file_info in missing_files[:5]:  # Show first 5 missing files
                            print(f"    - {file_info['name']} ({file_info['mime_type']}) - {file_info['source']}")
                        if len(missing_files) > 5:
                            print(f"    ... and {len(missing_files) - 5} more")
                else:
                    print("All available files were loaded successfully!")
            
            # Check for potential issues
            if len(docs) == 0:
                print("No documents found. Possible issues:")
                print("  - No supported file types in any accessible location")
                print("  - Permission issues with files")
                print("  - Files are too large or corrupted")
                print("  - Google Drive API permissions insufficient")
            
            elif len(docs) < 5:
                print("Few documents found. Consider:")
                print("  - Checking if files are in shared drives")
                print("  - Verifying file permissions")
                print("  - Ensuring files are supported types")
                print("  - Checking if files are in subfolders")
            
        else:
            print("No documents loaded from Google Drive")
            print("Troubleshooting suggestions:")
            print("  1. Check if you have files in any accessible Google Drive location")
            print("  2. Verify file types are supported (PDF, DOCX, TXT, etc.)")
            print("  3. Check file permissions and sharing settings")
            print("  4. Ensure Google Drive API has sufficient permissions")
            print("  5. Try the debug mode to see what files are accessible")
        
    except Exception as e:
        print(f"Error loading documents: {e}")

    return docs

def load_or_create_vectorstore(documents=None, user_id="default", use_existing=False, chunk_size=800, chunk_overlap=300):
    index_path = f"user_data/{user_id}/faiss_index"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if use_existing and os.path.exists(os.path.join(index_path, "index.faiss")):
        print("Loading existing FAISS index...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    if documents is None:
        raise ValueError("No documents provided and use_existing=False")

    print(f"Creating new FAISS index from documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
    
    # Enhanced text splitter with better parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]  # Better separators
    )
    
    texts = text_splitter.split_documents(documents)
    
    # Add chunk metadata for better retrieval
    for i, text in enumerate(texts):
        if hasattr(text, 'metadata'):
            text.metadata['chunk_id'] = i
            text.metadata['chunk_size'] = len(text.page_content)
            # Preserve original file information
            if 'file_name' not in text.metadata:
                text.metadata['file_name'] = text.metadata.get('source', 'Unknown')
    
    print(f"Created {len(texts)} text chunks")
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(index_path)

    return vectorstore

def build_faiss_index(documents, embeddings, index_dir):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"FAISS index saved at: {index_dir}")
    return vectorstore

def query_llm(vectorstore, query, evaluate_response=True, retrieval_count=8, model="claude-3-haiku-20240307", temperature=0.01, search_strategy="mmr", confidence_threshold=0.5, prompt_template="default", max_context_length=6000):
    def extract_unique_sentences(docs):
        import re
        from difflib import SequenceMatcher
        seen = set()
        section_pattern = re.compile(r'^(EXPERIENCE|EDUCATION|PROJECTS?|SKILLS?|ACHIEVEMENTS?|CERTIFICATIONS?|OBJECTIVE|SUMMARY|CONTACT|SOCIAL SERVICE|EXTRA-CURRICULAR ACTIVITIES|PROFESSIONAL EXPERIENCE)\b', re.IGNORECASE)
        placeholder_patterns = [
            re.compile(r'lorem ipsum', re.IGNORECASE),
            re.compile(r'company, location', re.IGNORECASE),
            re.compile(r'job title', re.IGNORECASE),
            re.compile(r'month 20xx', re.IGNORECASE),
            re.compile(r'n/a', re.IGNORECASE),
            re.compile(r'no relevant docs', re.IGNORECASE),
            re.compile(r'not provided', re.IGNORECASE),
            re.compile(r'no information', re.IGNORECASE),
            re.compile(r'♂phone', re.IGNORECASE),
            re.compile(r'/envel⌢pe', re.IGNORECASE),
            re.compile(r'/github', re.IGNORECASE),
            re.compile(r'/linkedin', re.IGNORECASE),
            re.compile(r'\+91\s*\d{10}', re.IGNORECASE),  # Phone numbers
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),  # Emails
            re.compile(r'https?://[^\s]+', re.IGNORECASE),  # URLs
            re.compile(r'^\d{4}\s*-\s*\w+\s*\d{4}$', re.IGNORECASE),  # Date ranges like "2021 - Mar 2023"
            re.compile(r'^\w+\s+\d{4}\s*-\s*\w+\s*\d{4}$', re.IGNORECASE),  # "JAN 2021 - Mar 2023"
        ]
        sections = {}
        current_section = None
        
        def is_similar_to_existing(text, existing_texts, threshold=0.8):
            """Check if text is semantically similar to any existing text"""
            text_lower = text.lower()
            for existing in existing_texts:
                existing_lower = existing.lower()
                # Check for high similarity
                similarity = SequenceMatcher(None, text_lower, existing_lower).ratio()
                if similarity > threshold:
                    return True
                # Check for key phrase overlap
                text_words = set(text_lower.split())
                existing_words = set(existing_lower.split())
                if len(text_words) > 3 and len(existing_words) > 3:
                    overlap = len(text_words.intersection(existing_words))
                    if overlap > min(len(text_words), len(existing_words)) * 0.7:
                        return True
            return False
        
        all_texts = []
        for doc in docs:
            # Split into lines for better section grouping
            lines = doc.page_content.splitlines()
            for line in lines:
                clean = line.strip()
                if len(clean) < 8:
                    continue
                # Skip placeholders
                if any(pat.search(clean) for pat in placeholder_patterns):
                    continue
                # Section header detection
                match = section_pattern.match(clean)
                if match:
                    current_section = match.group(1).upper()
                    if current_section not in sections:
                        sections[current_section] = []
                    continue
                # Check for duplicates and similarity
                if clean not in seen and not is_similar_to_existing(clean, all_texts):
                    if current_section:
                        sections.setdefault(current_section, []).append(clean)
                    else:
                        sections.setdefault('GENERAL', []).append(clean)
                    seen.add(clean)
                    all_texts.append(clean)
        return sections

    print(f"\nNew query received: {query}")
    print(f"Using model: {model}, retrieval_count: {retrieval_count}, temperature: {temperature}, search_strategy: {search_strategy}, confidence_threshold: {confidence_threshold}, prompt_template: {prompt_template}, max_context_length: {max_context_length}")
    
    # Enhanced retrieval with multiple strategies for better coverage
    all_docs = []
    
    # Increase retrieval count to search more of the vector database
    enhanced_retrieval_count = max(retrieval_count * 2, 12)  # At least 12 documents
    
    try:
        # Strategy 1: MMR for diverse results - search more documents
        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": enhanced_retrieval_count, "fetch_k": enhanced_retrieval_count * 3, "lambda_mult": 0.7}
        )
        mmr_docs = mmr_retriever.invoke(query)
        all_docs.extend(mmr_docs)
        print(f"MMR retrieved {len(mmr_docs)} documents from entire database")
    except Exception as e:
        print(f"MMR retrieval failed: {e}")
    
    try:
        # Strategy 2: Similarity search for most relevant - search more documents
        similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": enhanced_retrieval_count}
        )
        similarity_docs = similarity_retriever.invoke(query)
        all_docs.extend(similarity_docs)
        print(f"Similarity retrieved {len(similarity_docs)} documents from entire database")
    except Exception as e:
        print(f"Similarity retrieval failed: {e}")
    
    try:
        # Strategy 3: Similarity with lower threshold to get more documents
        threshold_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": enhanced_retrieval_count, "score_threshold": 0.1}  # Lower threshold
        )
        threshold_docs = threshold_retriever.invoke(query)
        all_docs.extend(threshold_docs)
        print(f"Threshold retrieved {len(threshold_docs)} documents from entire database")
    except Exception as e:
        print(f"Threshold retrieval failed: {e}")
    
    # Strategy 4: Additional broad search to ensure we don't miss anything
    try:
        broad_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}  # Very broad search
        )
        broad_docs = broad_retriever.invoke(query)
        all_docs.extend(broad_docs)
        print(f"Broad search retrieved {len(broad_docs)} documents from entire database")
    except Exception as e:
        print(f"Broad search failed: {e}")
    
    # Fallback: If no documents found, try basic similarity search
    if not all_docs:
        print("No documents found with advanced strategies, trying basic similarity search...")
        try:
            basic_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}  # Search more documents
            )
            all_docs = basic_retriever.invoke(query)
            print(f"Basic similarity retrieved {len(all_docs)} documents from entire database")
        except Exception as e:
            print(f"Basic similarity also failed: {e}")
            return "Error: Unable to retrieve any documents from the vector store."
    
    # Remove duplicates while preserving order
    seen_ids = set()
    unique_docs = []
    for doc in all_docs:
        # Use document ID or metadata for deduplication instead of content
        doc_id = doc.metadata.get('file_id', doc.metadata.get('id', doc.page_content[:100]))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
    
    print(f"Total unique documents after deduplication: {len(unique_docs)}")
    
    # Use ALL documents from the vector database - don't filter by relevance
    # This ensures we search the entire database and don't miss any information
    final_docs = unique_docs
    print(f"Using ALL {len(final_docs)} documents from vector database (no relevance filtering)")
    
    # Check if we have any documents at all
    if not final_docs:
        return "No documents found in the vector database. Please check if documents were properly loaded."
    
    # Limit to max_context_length while preserving most relevant
    total_chars = 0
    final_docs_limited = []
    for doc in final_docs:
        doc_length = len(doc.page_content)
        if total_chars + doc_length <= max_context_length * 2:  # Double the context length
            final_docs_limited.append(doc)
            total_chars += doc_length
        else:
            break
    
    final_docs = final_docs_limited
    print(f"Final documents used: {len(final_docs)} (total chars: {total_chars:,})")
    
    # Enhanced relevance filtering - focus on the specific person being asked about
    query_words = set(query.lower().split())
    relevant_docs = []
    
    # Extract potential names from query (words starting with capital letters)
    import re
    potential_names = re.findall(r'\b[A-Z][a-z]+\b', query)
    # Also check for lowercase names that might be the person's name
    lowercase_words = [word for word in query.lower().split() if len(word) > 3 and word not in ['tell', 'me', 'about', 'what', 'who', 'where', 'when', 'how', 'why']]
    potential_names.extend(lowercase_words)
    
    query_lower = query.lower()
    print(f"Query: '{query}'")
    print(f"Potential names found: {potential_names}")
    
    # If query contains a specific name, ONLY include documents with that name
    if potential_names:
        primary_name = potential_names[0].lower()
        print(f"Looking for documents about: {primary_name}")
        
        # STRICT filtering: only include documents that actually mention the person
        for doc in final_docs:
            doc_content_lower = doc.page_content.lower()
            doc_title_lower = doc.metadata.get('title', '').lower()
            
            # Check for exact name match in content or title
            name_in_content = primary_name in doc_content_lower
            name_in_title = primary_name in doc_title_lower
            
            # Only include if the name is actually found
            if name_in_content or name_in_title:
                relevant_docs.append(doc)
                print(f"Document '{doc.metadata.get('title', 'Unknown')}' contains '{primary_name}'")
            else:
                print(f"Document '{doc.metadata.get('title', 'Unknown')}' does NOT contain '{primary_name}' - EXCLUDED")
        
        print(f"Found {len(relevant_docs)} documents that actually mention '{primary_name}'")
        
        # If no documents found with the name, return early
        if not relevant_docs:
            return f"No documents found containing information about '{primary_name}'. Please check if this person's documents are in your Google Drive."
    else:
        print(f"No potential names found in query: '{query}'")
        # Fallback to word overlap if no specific name
        for doc in final_docs:
            doc_content_lower = doc.page_content.lower()
            doc_words = set(doc_content_lower.split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                relevant_docs.append(doc)
    
    if not relevant_docs:
        print(f"No documents found with strict relevance check for query: '{query}'")
        return f"No relevant documents found for your question. Please check if the information exists in your documents."
    
    # Use relevant documents
    final_docs = relevant_docs
    print(f"Using {len(final_docs)} relevant documents")
    
    # Debug: Show which documents are being used
    print("\nDocuments being used for answer:")
    for i, doc in enumerate(final_docs, 1):
        title = doc.metadata.get('title', 'Unknown Document')
        source = doc.metadata.get('source', 'Unknown Source')
        
        print(f"  {i}. File Name: {title}")
        print(f"     Source: {source}")
        print(f"     Content Length: {len(doc.page_content)} characters")
        print()
    
    # Debug: Show content vs metadata
    print("Content Analysis:")
    total_content_chars = sum(len(doc.page_content) for doc in final_docs)
    print(f"   Total content characters: {total_content_chars:,}")
    print(f"   Number of documents: {len(final_docs)}")
    print(f"   Average content per document: {total_content_chars // len(final_docs) if final_docs else 0:,} characters")
    
    # Check for documents with very little content
    low_content_docs = [doc for doc in final_docs if len(doc.page_content) < 50]
    if low_content_docs:
        print(f"  {len(low_content_docs)} documents have very little content (< 50 chars):")
        for doc in low_content_docs:
            print(f"    - {doc.metadata.get('title', 'Unknown')}: '{doc.page_content}'")
    
    # Enhanced prompt template to search entire vector database
    enhanced_prompt = """You are a helpful AI assistant. Answer the question by searching and using ONLY information that is EXPLICITLY stated in the documents provided in the context below. 

CRITICAL RULES - NO HALLUCINATION:
- ONLY use information that is EXPLICITLY written in the document text
- Do NOT add any information that is not directly stated in the documents
- Do NOT make assumptions or inferences beyond what is written
- If contact information is mentioned, provide it EXACTLY as written
- If links are mentioned, provide them EXACTLY as written
- If something is not clearly stated, say "This information is not provided in the documents"
- Do NOT summarize or interpret beyond the literal text
- Be extremely precise and factual
- When mentioning links, emails, phone numbers, etc., copy them EXACTLY as they appear

Context (DOCUMENTS): {context}

Question: {question}

Instructions:
1. Read through ALL documents carefully
2. Find information that EXPLICITLY answers the question
3. Quote or paraphrase the EXACT information from the documents
4. If information is missing or unclear, state "This information is not provided in the documents"
5. Do NOT add any information that is not in the documents
6. If links or contact details are mentioned, provide them exactly as written
7. Be very specific about what information is available vs. what is not

Answer:"""
    
    prompt = PromptTemplate(
        template=enhanced_prompt,
        input_variables=["context", "question"]
    )
    
    llm = ChatAnthropic(
        model=model,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=temperature
    )

    try:
        # Create a custom retriever that returns our combined documents
        from langchain.schema import BaseRetriever, Document
        from pydantic import Field
        
        class CombinedRetriever(BaseRetriever):
            documents: list = Field(default_factory=list)
            
            def __init__(self, documents):
                super().__init__()
                self.documents = documents
            
            def get_relevant_documents(self, query):
                return self.documents
            
            def invoke(self, query, config=None, **kwargs):
                return self.documents

        # Use our combined retriever
        combined_retriever = CombinedRetriever(final_docs)

        # Create QA chain with our retriever
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=combined_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        result = qa.invoke({"query": query})
        print("Claude answered successfully.")

        # Validate that the response is based on actual context
        response_text = result['result']

        # Create detailed source information with deduplication
        seen_sources = set()
        source_details = []
        for i, doc in enumerate(final_docs, 1):
            doc_title = doc.metadata.get('title', 'Unknown Document')
            doc_source = doc.metadata.get('source', 'Unknown Source')
            
            # Create unique source identifier
            source_id = f"{doc_title}|{doc_source}"
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)

            # Extract links from the document content
            import re
            links = re.findall(r'https?://[^\s]+', doc.page_content)

            source_details.append(f"{i}. {doc_title}")
            source_details.append(f"   Source: {doc_source}")
            if links:
                source_details.append(f"   Links: {', '.join(links)}")
            source_details.append("")

        sources = "\n".join(source_details)

        # Enhanced content validation
        context_validation = ""

        # Extract and highlight specific information from documents
        extracted_info = []
        for doc in final_docs:
            content = doc.page_content
            title = doc.metadata.get('title', 'Unknown Document')

            # Extract links
            import re
            links = re.findall(r'https?://[^\s]+', content)
            if links:
                extracted_info.append(f"Links in '{title}': {', '.join(links)}")

            # Extract emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails:
                extracted_info.append(f"Emails in '{title}': {', '.join(emails)}")

            # Extract phone numbers (more specific pattern)
            phones = re.findall(r'\+91\s*\d{10}', content)  # Indian phone numbers
            if phones:
                extracted_info.append(f"Phone numbers in '{title}': {', '.join(phones)}")

        if extracted_info:
            context_validation += "\n\nExtracted Information from Documents:\n" + "\n".join(extracted_info)

        # Check if response contains file names or titles
        file_names_in_response = []
        for doc in final_docs:
            doc_title = doc.metadata.get('title', '').lower()
            if doc_title and doc_title in response_text.lower():
                file_names_in_response.append(doc_title)
        
        if file_names_in_response:
            context_validation += f"\n\nWarning: Response may be based on file names rather than content. Detected file names: {', '.join(file_names_in_response)}"
        
        # Check if response contains specific details from content
        context_words = set()
        for doc in final_docs:
            # Get meaningful words from content (exclude common words)
            content_words = [word.lower() for word in doc.page_content.split() if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'they', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'about', 'many', 'then', 'them', 'these', 'some', 'what', 'into', 'more', 'only', 'very', 'just', 'know', 'take', 'than', 'first', 'been', 'good', 'much', 'over', 'think', 'also', 'here', 'most', 'even', 'make', 'life', 'still', 'should', 'through', 'back', 'after', 'work', 'years', 'never', 'become', 'under', 'same', 'another', 'family', 'seem', 'last', 'left', 'might', 'while', 'along', 'right', 'during', 'before', 'those', 'always', 'world', 'place', 'again', 'around', 'however', 'often', 'together', 'important', 'something', 'sometimes', 'nothing', 'everything', 'anything', 'someone', 'everyone', 'anyone', 'nobody', 'everybody', 'anybody']]
            context_words.update(content_words[:100])  # First 100 meaningful words
        
        response_words = set(response_text.lower().split())
        overlap = len(context_words.intersection(response_words))
        
        if overlap < 5:  # If very few content words overlap
            context_validation += "\n\nNote: This response may not be fully based on the actual document content. Please verify the information."
        
        # Check if response is too generic
        if len(response_text) < 100:
            context_validation += "\n\nNote: Response seems brief. This might indicate limited information in the documents or the model not using all available content."
        
        response = f"Answer:\n{response_text}{context_validation}\n\nSources Used:\n{sources}"
        
        # Add evaluation if requested
        if evaluate_response:
            try:
                from evaluator import evaluate_qa_response
                evaluation = evaluate_qa_response(query, result['result'])
                
                eval_info = f"\n\nResponse Quality:\n"
                eval_info += f"• Relevance: {evaluation.get('relevance_score', 'N/A')}/5\n"
                eval_info += f"• Completeness: {evaluation.get('completeness_score', 'N/A')}/5\n"
                eval_info += f"• Overall Score: {evaluation.get('overall_score', 'N/A')}/5"
                
                response += eval_info
                print(f"Response evaluated - Score: {evaluation.get('overall_score', 'N/A')}")
                
            except Exception as e:
                print(f"Evaluation failed: {e}")
                response += "\n\nEvaluation failed"
        
        return response
    except Exception as e:
        print(f"Claude query failed: {e}")
        return f"Error: {str(e)}"

def query_llm_stream(vectorstore, query, evaluate_response=True, retrieval_count=8, model="claude-3-haiku-20240307", temperature=0.01, search_strategy="mmr", confidence_threshold=0.5, prompt_template="default", max_context_length=6000):        
    """
    Streaming version of query_llm that yields tokens as they are generated.
    """
    print(f"\nNew streaming query received: {query}")
    print(f"Using model: {model}, retrieval_count: {retrieval_count}, temperature: {temperature}, search_strategy: {search_strategy}, confidence_threshold: {confidence_threshold}, prompt_template: {prompt_template}, max_context_length: {max_context_length}")
    
    # Enhanced retrieval with multiple strategies for better coverage
    all_docs = []
    
    # Increase retrieval count to search more of the vector database
    enhanced_retrieval_count = max(retrieval_count * 2, 12)  # At least 12 documents
    
    try:
        # Strategy 1: MMR for diverse results - search more documents
        mmr_retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": enhanced_retrieval_count, "fetch_k": enhanced_retrieval_count * 3, "lambda_mult": 0.7}
        )
        mmr_docs = mmr_retriever.invoke(query)
        all_docs.extend(mmr_docs)
        print(f"MMR retrieved {len(mmr_docs)} documents from entire database")
    except Exception as e:
        print(f"MMR retrieval failed: {e}")
    
    try:
        # Strategy 2: Similarity search for most relevant - search more documents
        similarity_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": enhanced_retrieval_count}
        )
        similarity_docs = similarity_retriever.invoke(query)
        all_docs.extend(similarity_docs)
        print(f"Similarity retrieved {len(similarity_docs)} documents from entire database")
    except Exception as e:
        print(f"Similarity retrieval failed: {e}")
    
    try:
        # Strategy 3: Similarity with lower threshold to get more documents
        threshold_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": enhanced_retrieval_count, "score_threshold": 0.1}  # Lower threshold
        )
        threshold_docs = threshold_retriever.invoke(query)
        all_docs.extend(threshold_docs)
        print(f"Threshold retrieved {len(threshold_docs)} documents from entire database")
    except Exception as e:
        print(f"Threshold retrieval failed: {e}")
    
    # Strategy 4: Additional broad search to ensure we don't miss anything
    try:
        broad_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20}  # Very broad search
        )
        broad_docs = broad_retriever.invoke(query)
        all_docs.extend(broad_docs)
        print(f"Broad search retrieved {len(broad_docs)} documents from entire database")
    except Exception as e:
        print(f"Broad search failed: {e}")
    
    # Fallback: If no documents found, try basic similarity search
    if not all_docs:
        print("No documents found with advanced strategies, trying basic similarity search...")
        try:
            basic_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 20}  # Search more documents
            )
            all_docs = basic_retriever.invoke(query)
            print(f"Basic similarity retrieved {len(all_docs)} documents from entire database")
        except Exception as e:
            print(f"Basic similarity also failed: {e}")
            yield "Error: Unable to retrieve any documents from the vector store."
            return
    
    # Remove duplicates while preserving order
    seen_ids = set()
    unique_docs = []
    for doc in all_docs:
        # Use document ID or metadata for deduplication instead of content
        doc_id = doc.metadata.get('file_id', doc.metadata.get('id', doc.page_content[:100]))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
    
    print(f"Total unique documents after deduplication: {len(unique_docs)}")
    
    # Use ALL documents from the vector database - don't filter by relevance
    # This ensures we search the entire database and don't miss any information
    final_docs = unique_docs
    print(f"Using ALL {len(final_docs)} documents from vector database (no relevance filtering)")
    
    # Check if we have any documents at all
    if not final_docs:
        yield "No documents found in the vector database. Please check if documents were properly loaded."
        return
    
    # Limit to max_context_length while preserving most relevant
    total_chars = 0
    final_docs_limited = []
    for doc in final_docs:
        doc_length = len(doc.page_content)
        if total_chars + doc_length <= max_context_length * 2:  # Double the context length
            final_docs_limited.append(doc)
            total_chars += doc_length
        else:
            break
    
    final_docs = final_docs_limited
    print(f"Final documents used: {len(final_docs)} (total chars: {total_chars:,})")
    
    # Enhanced relevance filtering - focus on the specific person being asked about
    query_words = set(query.lower().split())
    relevant_docs = []
    
    # Extract potential names from query (words starting with capital letters)
    import re
    potential_names = re.findall(r'\b[A-Z][a-z]+\b', query)
    # Also check for lowercase names that might be the person's name
    lowercase_words = [word for word in query.lower().split() if len(word) > 3 and word not in ['tell', 'me', 'about', 'what', 'who', 'where', 'when', 'how', 'why']]
    potential_names.extend(lowercase_words)
    
    query_lower = query.lower()
    print(f"Query: '{query}'")
    print(f"Potential names found: {potential_names}")
    
    # If query contains a specific name, ONLY include documents with that name
    if potential_names:
        primary_name = potential_names[0].lower()
        print(f"Looking for documents about: {primary_name}")
        
        # STRICT filtering: only include documents that actually mention the person
        for doc in final_docs:
            doc_content_lower = doc.page_content.lower()
            doc_title_lower = doc.metadata.get('title', '').lower()
            
            # Check for exact name match in content or title
            name_in_content = primary_name in doc_content_lower
            name_in_title = primary_name in doc_title_lower
            
            # Only include if the name is actually found
            if name_in_content or name_in_title:
                relevant_docs.append(doc)
                print(f"Document '{doc.metadata.get('title', 'Unknown')}' contains '{primary_name}'")
            else:
                print(f"Document '{doc.metadata.get('title', 'Unknown')}' does NOT contain '{primary_name}' - EXCLUDED")
        
        print(f"Found {len(relevant_docs)} documents that actually mention '{primary_name}'")
        
        # If no documents found with the name, return early
        if not relevant_docs:
            yield f"No documents found containing information about '{primary_name}'. Please check if this person's documents are in your Google Drive."
            return
    else:
        print(f"No potential names found in query: '{query}'")
        # Fallback to word overlap if no specific name
        for doc in final_docs:
            doc_content_lower = doc.page_content.lower()
            doc_words = set(doc_content_lower.split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                relevant_docs.append(doc)
    
    if not relevant_docs:
        print(f"No documents found with strict relevance check for query: '{query}'")
        yield f"No relevant documents found for your question. Please check if the information exists in your documents."
        return
    
    # Use relevant documents
    final_docs = relevant_docs
    print(f"Using {len(final_docs)} relevant documents")
    
    # Debug: Show which documents are being used
    print("\nDocuments being used for answer:")
    for i, doc in enumerate(final_docs, 1):
        title = doc.metadata.get('title', 'Unknown Document')
        source = doc.metadata.get('source', 'Unknown Source')
        
        print(f"  {i}. File Name: {title}")
        print(f"     Source: {source}")
        print(f"     Content Length: {len(doc.page_content)} characters")
        print()
    
    # Debug: Show content vs metadata
    print("Content Analysis:")
    total_content_chars = sum(len(doc.page_content) for doc in final_docs)
    print(f"   Total content characters: {total_content_chars:,}")
    print(f"   Number of documents: {len(final_docs)}")
    print(f"   Average content per document: {total_content_chars // len(final_docs) if final_docs else 0:,} characters")
    
    # Check for documents with very little content
    low_content_docs = [doc for doc in final_docs if len(doc.page_content) < 50]
    if low_content_docs:
        print(f"  {len(low_content_docs)} documents have very little content (< 50 chars):")
        for doc in low_content_docs:
            print(f"    - {doc.metadata.get('title', 'Unknown')}: '{doc.page_content}'")
    
    # Enhanced prompt template to search entire vector database
    enhanced_prompt = """You are a helpful AI assistant. Answer the question by searching and using ONLY information that is EXPLICITLY stated in the documents provided in the context below. 

CRITICAL RULES - NO HALLUCINATION:
- ONLY use information that is EXPLICITLY written in the document text
- Do NOT add any information that is not directly stated in the documents
- Do NOT make assumptions or inferences beyond what is written
- If contact information is mentioned, provide it EXACTLY as written
- If links are mentioned, provide them EXACTLY as written
- If something is not clearly stated, say "This information is not provided in the documents"
- Do NOT summarize or interpret beyond the literal text
- Be extremely precise and factual
- When mentioning links, emails, phone numbers, etc., copy them EXACTLY as they appear

Context (DOCUMENTS): {context}

Question: {question}

Instructions:
1. Read through ALL documents carefully
2. Find information that EXPLICITLY answers the question
3. Quote or paraphrase the EXACT information from the documents
4. If information is missing or unclear, state "This information is not provided in the documents"
5. Do NOT add any information that is not in the documents
6. If links or contact details are mentioned, provide them exactly as written
7. Be very specific about what information is available vs. what is not

Answer:"""
    
    prompt = PromptTemplate(
        template=enhanced_prompt,
        input_variables=["context", "question"]
    )
    
    llm = ChatAnthropic(
        model=model,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=temperature
    )

    try:
        # Create a custom retriever that returns our combined documents
        from langchain.schema import BaseRetriever, Document
        from pydantic import Field
        
        class CombinedRetriever(BaseRetriever):
            documents: list = Field(default_factory=list)
            
            def __init__(self, documents):
                super().__init__()
                self.documents = documents
            
            def get_relevant_documents(self, query):
                return self.documents
            
            def invoke(self, query, config=None, **kwargs):
                return self.documents

        # Use our combined retriever
        combined_retriever = CombinedRetriever(final_docs)

        # Create QA chain with our retriever
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=combined_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        # Stream the response
        result = qa.invoke({"query": query})
        response_text = result['result']
        
        # Create detailed source information with deduplication
        seen_sources = set()
        source_details = []
        for i, doc in enumerate(final_docs, 1):
            doc_title = doc.metadata.get('title', 'Unknown Document')
            doc_source = doc.metadata.get('source', 'Unknown Source')
            
            # Create unique source identifier
            source_id = f"{doc_title}|{doc_source}"
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)

            # Extract links from the document content
            import re
            links = re.findall(r'https?://[^\s]+', doc.page_content)

            source_details.append(f"{i}. {doc_title}")
            source_details.append(f"   Source: {doc_source}")
            if links:
                source_details.append(f"   Links: {', '.join(links)}")
            source_details.append("")

        sources = "\n".join(source_details)
        
        # Enhanced content validation
        context_validation = ""

        # Extract and highlight specific information from documents
        extracted_info = []
        for doc in final_docs:
            content = doc.page_content
            title = doc.metadata.get('title', 'Unknown Document')

            # Extract links
            import re
            links = re.findall(r'https?://[^\s]+', content)
            if links:
                extracted_info.append(f"Links in '{title}': {', '.join(links)}")

            # Extract emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
            if emails:
                extracted_info.append(f"Emails in '{title}': {', '.join(emails)}")

            # Extract phone numbers (more specific pattern)
            phones = re.findall(r'\+91\s*\d{10}', content)  # Indian phone numbers
            if phones:
                extracted_info.append(f"Phone numbers in '{title}': {', '.join(phones)}")

        if extracted_info:
            context_validation += "\n\nExtracted Information from Documents:\n" + "\n".join(extracted_info)

        # Check if response contains file names or titles
        file_names_in_response = []
        for doc in final_docs:
            doc_title = doc.metadata.get('title', '').lower()
            if doc_title and doc_title in response_text.lower():
                file_names_in_response.append(doc_title)
        
        if file_names_in_response:
            context_validation += f"\n\nWarning: Response may be based on file names rather than content. Detected file names: {', '.join(file_names_in_response)}"
        
        # Check if response contains specific details from content
        context_words = set()
        for doc in final_docs:
            # Get meaningful words from content (exclude common words)
            content_words = [word.lower() for word in doc.page_content.split() if len(word) > 3 and word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'they', 'will', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'about', 'many', 'then', 'them', 'these', 'some', 'what', 'into', 'more', 'only', 'very', 'just', 'know', 'take', 'than', 'first', 'been', 'good', 'much', 'over', 'think', 'also', 'here', 'most', 'even', 'make', 'life', 'still', 'should', 'through', 'back', 'after', 'work', 'years', 'never', 'become', 'under', 'same', 'another', 'family', 'seem', 'last', 'left', 'might', 'while', 'along', 'right', 'during', 'before', 'those', 'always', 'world', 'place', 'again', 'around', 'however', 'often', 'together', 'important', 'something', 'sometimes', 'nothing', 'everything', 'anything', 'someone', 'everyone', 'anyone', 'nobody', 'everybody', 'anybody']]
            context_words.update(content_words[:100])  # First 100 meaningful words
        
        response_words = set(response_text.lower().split())
        overlap = len(context_words.intersection(response_words))
        
        if overlap < 5:  # If very few content words overlap
            context_validation += "\n\nNote: This response may not be fully based on the actual document content. Please verify the information."
        
        # Check if response is too generic
        if len(response_text) < 100:
            context_validation += "\n\nNote: Response seems brief. This might indicate limited information in the documents or the model not using all available content."
        
        response = f"Answer:\n{response_text}{context_validation}\n\nSources Used:\n{sources}"
        
        # Add evaluation if requested
        if evaluate_response:
            try:
                from evaluator import evaluate_qa_response
                evaluation = evaluate_qa_response(query, result['result'])
                
                eval_info = f"\n\nResponse Quality:\n"
                eval_info += f"• Relevance: {evaluation.get('relevance_score', 'N/A')}/5\n"
                eval_info += f"• Completeness: {evaluation.get('completeness_score', 'N/A')}/5\n"
                eval_info += f"• Overall Score: {evaluation.get('overall_score', 'N/A')}/5"
                
                response += eval_info
                print(f"Response evaluated - Score: {evaluation.get('overall_score', 'N/A')}")
                
            except Exception as e:
                print(f"Evaluation failed: {e}")
                response += "\n\nEvaluation failed"
        
        # Stream the response character by character for a more natural effect
        for i in range(len(response)):
            yield response[:i+1]
            
    except Exception as e:
        print(f"Claude query failed: {e}")
        yield f"Error: {str(e)}"
