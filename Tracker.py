import discord
from discord.ext import commands
import json
import os
from datetime import datetime, timedelta
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.styles import Font
from dotenv import load_dotenv
import asyncio
from collections import defaultdict
import logging
import hashlib
import jsonschema
from pathlib import Path
import ijson
import gc
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Dict, List, Any, Optional

# Load .env file
load_dotenv()

# Setup secure logging
logging.basicConfig(
    level=logging.DEBUG,  # Debug level açıldı
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_security.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Bot settings
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.guilds = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Secure file paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_FILE = BASE_DIR / 'message_tracking.json'
CONTENT_FILE = BASE_DIR / 'content_tracking.json'
CONFIG_FILE = BASE_DIR / 'bot_config.json'
AUTH_CONFIG_FILE = BASE_DIR / 'authorized_config.json'

# JSON schemas for validation
USER_DATA_SCHEMA = {
    "type": "object",
    "properties": {
        "username": {"type": "string", "maxLength": 100},
        "current_roles": {"type": "array", "items": {"type": "string", "maxLength": 100}},
        "messages": {"type": "array"},
        "contents": {"type": "array"}
    },
    "required": ["username", "current_roles", "messages"]
}

CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "content_channel_ids": {"type": "array", "items": {"type": "integer"}}
    }
}

# Permission system - Only users with these IDs can use commands
AUTHORIZED_USERS = []

# Permission system - Users with these role IDs can use commands
AUTHORIZED_ROLE_IDS = []

# Message tracking - Users with these role IDs will have their messages tracked
TRACKED_ROLE_IDS = []

# Content tracking - Channel IDs for content tracking (Twitter/X links) - supports multiple channels
CONTENT_CHANNEL_IDS = []  # Will be loaded from config file

# Rate limiting system
COMMAND_COOLDOWN = 5  # seconds
user_command_times = defaultdict(lambda: 0)

# Maximum number of authorized users/roles for security
MAX_AUTHORIZED_USERS = 50
MAX_AUTHORIZED_ROLES = 20

# Performance optimization settings for large datasets
CHUNK_SIZE = 10000  # Process users in chunks of 10k
MAX_MEMORY_USERS = 50000  # Keep max 50k users in memory at once
ENABLE_PARALLEL_PROCESSING = True  # Enable multi-threading
MAX_WORKER_THREADS = 4  # Number of worker threads

# Time filters (in seconds) - added 'total' for all-time statistics
TIME_FILTERS = {
    'total': 0,  # 0 means no time limit (all messages)
    '24h': 24 * 60 * 60,
    '7d': 7 * 24 * 60 * 60,
    '30d': 30 * 24 * 60 * 60,
    '90d': 90 * 24 * 60 * 60,
    '180d': 180 * 24 * 60 * 60,
    '360d': 360 * 24 * 60 * 60
}

def has_permission(ctx):
    """Check if user has permission to use commands"""
    # Rate limit check
    if not check_rate_limit(ctx.author.id):
        return False

    # Authorized user check
    if ctx.author.id in AUTHORIZED_USERS:
        return True

    # Authorized role ID check
    user_role_ids = [role.id for role in ctx.author.roles]
    for role_id in user_role_ids:
        if role_id in AUTHORIZED_ROLE_IDS:
            return True

    return False

async def silent_delete(ctx):
    """Silently delete command (unauthorized usage)"""
    try:
        await ctx.message.delete()
    except discord.NotFound:
        # Message already deleted
        pass
    except discord.Forbidden:
        # No permission to delete
        logger.warning(f"No permission to delete message from {ctx.author.id}")
    except Exception as e:
        logger.error(f"Unexpected error deleting message: {e}")

def extract_twitter_links(content):
    """Extract and normalize Twitter/X status links from message content.

    - Accepts twitter.com and x.com status URLs (with or without query/fragment)
    - Normalizes to https://x.com/{username}/status/{id}
    - Ignores non-status or domain-only links
    - Deduplicates within the message
    """
    import re
    from urllib.parse import urlparse

    def normalize_x_twitter_link(url):
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return None
            host = parsed.netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            if host not in ("x.com", "twitter.com"):
                return None
            path = parsed.path.strip("/")
            parts = path.split("/")
            if len(parts) < 3:
                return None
            username, token, status_id = parts[0], parts[1].lower(), parts[2]
            if token != "status":
                return None
            if not username or not status_id.isdigit():
                return None
            # Canonical form without query/fragment
            return f"https://x.com/{username}/status/{status_id}"
        except Exception:
            return None

    # Match potential status URLs (allowing optional query part)
    pattern = r'https?://(?:www\.)?(?:twitter\.com|x\.com)/[A-Za-z0-9_]+/status/\d+(?:\?[^\s]*)?'

    normalized_links = set()
    for raw_url in re.findall(pattern, content):
        normalized = normalize_x_twitter_link(raw_url)
        if normalized:
            normalized_links.add(normalized)

    return list(normalized_links)

def load_content_data():
    """Load content data from JSON file securely"""
    return secure_file_read(CONTENT_FILE)

def save_content_data(data):
    """Save content data to JSON file securely"""
    if not secure_file_write(CONTENT_FILE, data):
        logger.error("Failed to save content data")

def load_config():
    """Load bot configuration from JSON file with validation"""
    global CONTENT_CHANNEL_IDS
    try:
        config = secure_file_read(CONFIG_FILE, CONFIG_SCHEMA)
        if config:
            CONTENT_CHANNEL_IDS = config.get('content_channel_ids', [])
            logger.info(f"Loaded {len(CONTENT_CHANNEL_IDS)} content channels from config")
    except Exception as e:
        logger.error(f"Error loading config: {e}")

def save_config():
    """Save bot configuration to JSON file securely"""
    try:
        config = {
            'content_channel_ids': CONTENT_CHANNEL_IDS
        }
        if not secure_file_write(CONFIG_FILE, config):
            logger.error("Failed to save config")
    except Exception as e:
        logger.error(f"Error saving config: {e}")

def is_valid_channel_id(channel_id_str):
    """Check if it's a valid Discord channel ID with enhanced validation"""
    try:
        if not isinstance(channel_id_str, str):
            return False
            
        # Remove whitespace and check for non-digit characters
        channel_id_str = channel_id_str.strip()
        if not channel_id_str.isdigit():
            return False
            
        channel_id = int(channel_id_str)
        
        # Discord snowflake validation
        if not (17 <= len(str(channel_id)) <= 19):
            return False
            
        # Check if it's a valid Discord snowflake (after Discord epoch)
        discord_epoch = 1420070400000  # January 1, 2015
        timestamp = (channel_id >> 22) + discord_epoch
        current_timestamp = datetime.now().timestamp() * 1000
        
        return discord_epoch <= timestamp <= current_timestamp
        
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Invalid channel ID validation attempt: {channel_id_str}, error: {e}")
        return False

def check_rate_limit(user_id):
    """Check rate limit"""
    current_time = asyncio.get_event_loop().time()
    last_command_time = user_command_times[user_id]

    if current_time - last_command_time < COMMAND_COOLDOWN:
        return False

    user_command_times[user_id] = current_time
    return True

def is_valid_user_id(user_id_str):
    """Check if it's a valid Discord user ID with enhanced validation"""
    try:
        if not isinstance(user_id_str, str):
            return False
        
        # Remove whitespace and check for non-digit characters
        user_id_str = user_id_str.strip()
        if not user_id_str.isdigit():
            return False
            
        user_id = int(user_id_str)
        
        # Discord snowflake validation
        # Must be between 17-19 digits and within valid timestamp range
        if not (17 <= len(str(user_id)) <= 19):
            return False
            
        # Check if it's a valid Discord snowflake (after Discord epoch)
        discord_epoch = 1420070400000  # January 1, 2015
        timestamp = (user_id >> 22) + discord_epoch
        current_timestamp = datetime.now().timestamp() * 1000
        
        return discord_epoch <= timestamp <= current_timestamp
        
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Invalid user ID validation attempt: {user_id_str}, error: {e}")
        return False

def is_valid_role_id(role_id_str):
    """Check if it's a valid Discord role ID with enhanced validation"""
    try:
        if not isinstance(role_id_str, str):
            return False
            
        # Remove whitespace and check for non-digit characters
        role_id_str = role_id_str.strip()
        if not role_id_str.isdigit():
            return False
            
        role_id = int(role_id_str)
        
        # Discord snowflake validation
        if not (17 <= len(str(role_id)) <= 19):
            return False
            
        # Check if it's a valid Discord snowflake (after Discord epoch)
        discord_epoch = 1420070400000  # January 1, 2015
        timestamp = (role_id >> 22) + discord_epoch
        current_timestamp = datetime.now().timestamp() * 1000
        
        return discord_epoch <= timestamp <= current_timestamp
        
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Invalid role ID validation attempt: {role_id_str}, error: {e}")
        return False

def validate_json_data(data, schema):
    """Validate JSON data against schema"""
    try:
        jsonschema.validate(data, schema)
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"JSON validation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"JSON validation error: {e}")
        return False

def secure_file_write(file_path, data):
    """Securely write data to file with backup"""
    try:
        # Create backup if file exists
        if file_path.exists():
            backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
            file_path.rename(backup_path)
        
        # Write to temporary file first
        temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp")
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Atomic move
        temp_path.rename(file_path)
        logger.info(f"Successfully wrote data to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing to {file_path}: {e}")
        return False

def secure_file_read(file_path, schema=None):
    """Securely read and validate JSON file"""
    try:
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if schema and not validate_json_data(data, schema):
            logger.error(f"Invalid data structure in {file_path}")
            return {}
            
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return {}

async def save_authorized_data():
    """Save permission data to JSON with validation"""
    data = {
        'authorized_users': AUTHORIZED_USERS,
        'authorized_role_ids': AUTHORIZED_ROLE_IDS
    }

    if not secure_file_write(AUTH_CONFIG_FILE, data):
        logger.error("Failed to save authorized data")

async def load_authorized_data():
    """Load permission data from JSON and environment variables with enhanced security"""
    global AUTHORIZED_USERS, AUTHORIZED_ROLE_IDS, TRACKED_ROLE_IDS

    # Load from environment variables first (higher priority)
    env_authorized_users = os.getenv('AUTHORIZED_USERS', '')
    env_tracked_roles = os.getenv('TRACKED_ROLE_IDS', '')
    
    if env_authorized_users:
        try:
            # Enhanced validation for environment variables
            user_ids = []
            for user_id_str in env_authorized_users.split(','):
                user_id_str = user_id_str.strip()
                if user_id_str and is_valid_user_id(user_id_str):
                    user_ids.append(int(user_id_str))
                elif user_id_str:
                    logger.warning(f"Invalid user ID in AUTHORIZED_USERS: {user_id_str}")
            AUTHORIZED_USERS = user_ids
            logger.info(f"Loaded {len(user_ids)} authorized users from environment")
        except Exception as e:
            logger.error(f"Error parsing AUTHORIZED_USERS from .env: {e}")
    
    if env_tracked_roles:
        try:
            # Enhanced validation for environment variables
            role_ids = []
            for role_id_str in env_tracked_roles.split(','):
                role_id_str = role_id_str.strip()
                if role_id_str and is_valid_role_id(role_id_str):
                    role_ids.append(int(role_id_str))
                elif role_id_str:
                    logger.warning(f"Invalid role ID in TRACKED_ROLE_IDS: {role_id_str}")
            TRACKED_ROLE_IDS = role_ids
            logger.info(f"Loaded {len(role_ids)} tracked roles from environment")
        except Exception as e:
            logger.error(f"Error parsing TRACKED_ROLE_IDS from .env: {e}")

    # Load from JSON file (fallback or additional data)
    data = secure_file_read(AUTH_CONFIG_FILE)
    if data:
        # Only load from JSON if not already loaded from env
        if not AUTHORIZED_USERS:
            AUTHORIZED_USERS = data.get('authorized_users', [])
        if not AUTHORIZED_ROLE_IDS:
            AUTHORIZED_ROLE_IDS = data.get('authorized_role_ids', [])

def has_owner_permission(ctx):
    """Only bot owner can manage permissions with enhanced security"""
    try:
        # Get bot owner ID from environment or application info
        owner_id = os.getenv('BOT_OWNER_ID')
        if owner_id:
            if not is_valid_user_id(owner_id):
                logger.error(f"Invalid BOT_OWNER_ID in environment: {owner_id}")
                return False
            return ctx.author.id == int(owner_id)
        
        # Fallback to bot owner_id if available
        if ctx.bot.owner_id:
            return ctx.author.id == ctx.bot.owner_id
            
        # If no owner is set, deny access and log security event
        logger.critical(f"No bot owner configured! Access denied for user {ctx.author.id}")
        return False
        
    except Exception as e:
        logger.error(f"Error checking owner permission: {e}")
        return False

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors securely"""
    if isinstance(error, commands.CheckFailure):
        # Permission error - silently delete message and log security event
        logger.warning(f"Unauthorized command attempt by {ctx.author.id} ({ctx.author.name}): {ctx.message.content}")
        await silent_delete(ctx)
    elif isinstance(error, commands.CommandNotFound):
        # Command not found - silently ignore
        pass
    elif isinstance(error, commands.MissingRequiredArgument):
        # Missing argument - log but don't expose details
        logger.info(f"Missing argument in command by {ctx.author.id}: {ctx.command}")
    else:
        # Other errors - log securely without exposing sensitive information
        error_hash = hashlib.md5(str(error).encode()).hexdigest()[:8]
        logger.error(f"Command error [{error_hash}]: {type(error).__name__}")
        logger.debug(f"Full error details [{error_hash}]: {error}")

def load_data():
    """Load data from JSON file with validation"""
    return secure_file_read(DATA_FILE)

def save_data(data):
    """Save data to JSON file securely"""
    if not secure_file_write(DATA_FILE, data):
        logger.error("Failed to save message tracking data")

def stream_user_data(file_path: Path, chunk_size: int = CHUNK_SIZE):
    """Stream user data in chunks to avoid memory issues"""
    try:
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return
            
        with open(file_path, 'rb') as file:
            # Use ijson to stream parse the JSON file
            parser = ijson.parse(file)
            current_user_id = None
            current_user_data = {}
            users_processed = 0
            chunk = {}
            
            for prefix, event, value in parser:
                if prefix == '' and event == 'map_key':
                    # New user ID
                    if current_user_id and current_user_data:
                        chunk[current_user_id] = current_user_data
                        users_processed += 1
                        
                        # Yield chunk when it reaches chunk_size
                        if users_processed >= chunk_size:
                            yield chunk
                            chunk = {}
                            users_processed = 0
                            gc.collect()  # Force garbage collection
                    
                    current_user_id = value
                    current_user_data = {}
                    
                elif current_user_id:
                    # Parse user data
                    if prefix.endswith('.username') and event == 'string':
                        current_user_data['username'] = value
                    elif prefix.endswith('.current_roles') and event == 'start_array':
                        current_user_data['current_roles'] = []
                    elif prefix.endswith('.current_roles.item') and event == 'string':
                        if 'current_roles' not in current_user_data:
                            current_user_data['current_roles'] = []
                        current_user_data['current_roles'].append(value)
                    elif prefix.endswith('.messages') and event == 'start_array':
                        current_user_data['messages'] = []
                    elif prefix.endswith('.contents') and event == 'start_array':
                        current_user_data['contents'] = []
                    elif '.messages.' in prefix and event in ['string', 'number']:
                        # Parse message data
                        if 'messages' not in current_user_data:
                            current_user_data['messages'] = []
                        # This is simplified - you may need to handle message objects properly
                    elif '.contents.' in prefix and event in ['string', 'number']:
                        # Parse content data
                        if 'contents' not in current_user_data:
                            current_user_data['contents'] = []
            
            # Yield remaining data
            if current_user_id and current_user_data:
                chunk[current_user_id] = current_user_data
            if chunk:
                yield chunk
                
    except Exception as e:
        logger.error(f"Error streaming data from {file_path}: {e}")

def process_user_chunk(chunk: Dict[str, Any], filters: Dict[str, Any], 
                      time_periods: Dict[str, datetime], role_conditions: Dict[str, List[str]],
                      ctx) -> List[Dict[str, Any]]:
    """Process a chunk of users for filtering - optimized for parallel processing"""
    results = []
    
    for user_id, user_data in chunk.items():
        try:
            # Pre-calculate all time period counts in one pass
            all_message_counts = {}
            content_counts = {}
            
            # Initialize counts
            for period_name in time_periods.keys():
                all_message_counts[period_name] = 0
                content_counts[period_name] = 0
            
            # Single pass through messages
            channel_message_counts = {}
            for period_name in time_periods.keys():
                channel_message_counts[period_name] = 0
                
            for msg in user_data.get('messages', []):
                try:
                    msg_time = datetime.fromisoformat(msg['timestamp'])
                    
                    # Check if message is in the filtered channel
                    msg_in_channel = True
                    if 'channel' in filters:
                        msg_in_channel = msg.get('channel_name', '').lower() == filters['channel'].lower()
                    elif 'channel_id' in filters:
                        msg_in_channel = str(msg.get('channel_id', '')) == str(filters['channel_id'])
                    
                    for period_name, period_start in time_periods.items():
                        if msg_time > period_start:
                            all_message_counts[period_name] += 1
                            if msg_in_channel:
                                channel_message_counts[period_name] += 1
                except Exception:
                    continue
            
            # Single pass through contents
            for content_entry in user_data.get('contents', []):
                try:
                    content_time = datetime.fromisoformat(content_entry['timestamp'])
                    for period_name, period_start in time_periods.items():
                        if content_time > period_start:
                            content_counts[period_name] += len(content_entry.get('twitter_links', []))
                except Exception:
                    continue
            
            # Apply filters efficiently
            should_include = True
            
            # Get the target time period count
            target_period = filters.get('time_period', 'total')
            
            # Use channel-specific message count if channel filter is active
            if 'channel' in filters or 'channel_id' in filters:
                message_count = channel_message_counts.get(target_period, 0)
            else:
                message_count = all_message_counts.get(target_period, 0)
            
            content_count = content_counts.get(target_period, 0)
            
            # Message filter
            if 'message' in filters and message_count <= filters['message']:
                should_include = False
                
            # Content filter
            if 'content' in filters and content_count <= filters['content']:
                should_include = False
            
            # Role filtering (if needed)
            if should_include and (role_conditions['must_have'] or role_conditions['any_of'] or role_conditions['must_not_have']):
                should_include = check_user_roles(user_id, role_conditions, ctx)
            
            if should_include:
                results.append({
                    'user_id': user_id,
                    'username': user_data.get('username', 'Unknown'),
                    'roles': user_data.get('current_roles', []),
                    'message_count': message_count,
                    'content_count': content_count,
                    'all_message_counts': all_message_counts,
                    'channel_message_counts': channel_message_counts,
                    'all_content_counts': content_counts,
                    'channels': list(set([msg.get('channel_name', 'Unknown') for msg in user_data.get('messages', [])])),
                    'x_links': []  # Will be populated if needed
                })
                
        except Exception as e:
            logger.warning(f"Error processing user {user_id}: {e}")
            continue
    
    return results

def check_user_roles(user_id: str, role_conditions: Dict[str, List[str]], ctx) -> bool:
    """Check if user meets role conditions - optimized version"""
    try:
        # This is expensive - cache results or use async version
        member = ctx.guild.get_member(int(user_id))
        if not member:
            logger.debug(f"Member {user_id} not found in guild for role check")
            return False
            
        user_role_ids = [role.id for role in member.roles]
        logger.debug(f"User {user_id} has role IDs: {user_role_ids}")
        
        # Check MUST HAVE roles (AND logic)
        for role_id in role_conditions['must_have']:
            try:
                role_id_int = int(role_id)
                if role_id_int not in user_role_ids:
                    logger.debug(f"User {user_id} missing required role {role_id}")
                    return False
                else:
                    logger.debug(f"User {user_id} has required role {role_id}")
            except ValueError:
                logger.warning(f"Invalid role ID format: {role_id}")
                return False
        
        # Check ANY OF roles (OR logic)
        if role_conditions['any_of']:
            has_any = False
            for role_id in role_conditions['any_of']:
                try:
                    role_id_int = int(role_id)
                    if role_id_int in user_role_ids:
                        has_any = True
                        logger.debug(f"User {user_id} has any_of role {role_id}")
                        break
                except ValueError:
                    continue
            if not has_any:
                logger.debug(f"User {user_id} missing any_of roles: {role_conditions['any_of']}")
                return False
        
        # Check MUST NOT HAVE roles (NOT logic)
        for role_id in role_conditions['must_not_have']:
            try:
                role_id_int = int(role_id)
                if role_id_int in user_role_ids:
                    logger.debug(f"User {user_id} has forbidden role {role_id}")
                    return False
            except ValueError:
                continue
        
        logger.debug(f"User {user_id} passes all role checks")        
        return True
        
    except Exception as e:
        logger.warning(f"Error checking roles for user {user_id}: {e}")
        return False

async def process_large_dataset_streaming(ctx, filters, all_time_periods, role_conditions, time_limit):
    """Process large datasets using streaming and parallel processing"""
    filtered_results = []
    total_processed = 0
    
    # Send progress message
    progress_msg = await ctx.send("🔄 Processing large dataset... 0 users processed")
    
    try:
        if ENABLE_PARALLEL_PROCESSING:
            # Use ThreadPoolExecutor for CPU-intensive processing
            with ThreadPoolExecutor(max_workers=MAX_WORKER_THREADS) as executor:
                futures = []
                
                # Process data in chunks using streaming
                for chunk in stream_user_data(DATA_FILE, CHUNK_SIZE):
                    if not chunk:
                        continue
                    
                    # Submit chunk for parallel processing
                    future = executor.submit(process_user_chunk, chunk, filters, all_time_periods, role_conditions, ctx)
                    futures.append(future)
                    
                    # Process completed futures to avoid memory buildup
                    if len(futures) >= MAX_WORKER_THREADS:
                        for future in futures:
                            chunk_results = future.result()
                            filtered_results.extend(chunk_results)
                            total_processed += len(chunk)
                        futures = []
                        
                        # Update progress
                        try:
                            await progress_msg.edit(content=f"🔄 Processing large dataset... {total_processed:,} users processed")
                        except:
                            pass
                        
                        gc.collect()  # Force garbage collection
                
                # Process remaining futures
                for future in futures:
                    chunk_results = future.result()
                    filtered_results.extend(chunk_results)
                    total_processed += len(chunk)
        else:
            # Sequential processing if parallel processing is disabled
            for chunk in stream_user_data(DATA_FILE, CHUNK_SIZE):
                if not chunk:
                    continue
                    
                chunk_results = process_user_chunk(chunk, filters, all_time_periods, role_conditions, ctx)
                filtered_results.extend(chunk_results)
                total_processed += len(chunk)
                
                # Update progress every 50k users
                if total_processed % 50000 == 0:
                    try:
                        await progress_msg.edit(content=f"🔄 Processing large dataset... {total_processed:,} users processed")
                    except:
                        pass
                
                gc.collect()
        
        # Delete progress message
        try:
            await progress_msg.delete()
        except:
            pass
            
        logger.info(f"Streaming processing completed: {total_processed:,} users processed, {len(filtered_results)} results")
        return filtered_results
        
    except Exception as e:
        logger.error(f"Error in streaming processing: {e}")
        try:
            await progress_msg.delete()
        except:
            pass
        return []

def process_standard_dataset(data, filters, all_time_periods, role_conditions, time_limit, ctx):
    """Process standard datasets (non-streaming) - optimized version"""
    filtered_results = []
    
    logger.info(f"Processing {len(data)} users with standard method")
    
    for user_id, user_data in data.items():
        try:
            # Pre-calculate all time period counts in one pass
            all_message_counts = {}
            content_counts = {}
            
            # Initialize counts
            for period_name in all_time_periods.keys():
                all_message_counts[period_name] = 0
                content_counts[period_name] = 0
            
            # Single pass through messages
            channel_message_counts = {}
            for period_name in all_time_periods.keys():
                channel_message_counts[period_name] = 0
                
            for msg in user_data.get('messages', []):
                try:
                    msg_time = datetime.fromisoformat(msg['timestamp'])
                    
                    # Check if message is in the filtered channel
                    msg_in_channel = True
                    if 'channel' in filters:
                        msg_in_channel = msg.get('channel_name', '').lower() == filters['channel'].lower()
                    elif 'channel_id' in filters:
                        msg_in_channel = str(msg.get('channel_id', '')) == str(filters['channel_id'])
                    
                    for period_name, period_start in all_time_periods.items():
                        if msg_time > period_start:
                            all_message_counts[period_name] += 1
                            if msg_in_channel:
                                channel_message_counts[period_name] += 1
                except Exception:
                    continue
            
            # Single pass through contents
            for content_entry in user_data.get('contents', []):
                try:
                    content_time = datetime.fromisoformat(content_entry['timestamp'])
                    for period_name, period_start in all_time_periods.items():
                        if content_time > period_start:
                            content_counts[period_name] += len(content_entry.get('twitter_links', []))
                except Exception:
                    continue
            
            # Apply filters efficiently
            should_include = True
            
            # Get the target time period count
            target_period = filters.get('time_period', 'total')
            
            # Use channel-specific message count if channel filter is active
            if 'channel' in filters or 'channel_id' in filters:
                message_count = channel_message_counts.get(target_period, 0)
            else:
                message_count = all_message_counts.get(target_period, 0)
            
            content_count = content_counts.get(target_period, 0)
            
            # Message filter
            if 'message' in filters and message_count <= filters['message']:
                should_include = False
                
            # Content filter
            if 'content' in filters and content_count <= filters['content']:
                should_include = False
            
            # Role filtering (if needed)
            if should_include and (role_conditions['must_have'] or role_conditions['any_of'] or role_conditions['must_not_have']):
                should_include = check_user_roles(user_id, role_conditions, ctx)
            
            if should_include:
                # Collect X links for the specified time period (sorted by newest first)
                x_links_in_period = []
                try:
                    link_to_time = {}
                    for entry in user_data.get('contents', []):
                        try:
                            entry_time = datetime.fromisoformat(entry.get('timestamp', datetime.min.isoformat()))
                        except Exception:
                            continue
                        if entry_time > time_limit:
                            for link in entry.get('twitter_links', []):
                                if (link not in link_to_time) or (entry_time > link_to_time[link]):
                                    link_to_time[link] = entry_time
                    x_links_in_period = [lt[0] for lt in sorted(link_to_time.items(), key=lambda kv: kv[1], reverse=True)]
                except Exception:
                    x_links_in_period = []
                
                filtered_results.append({
                    'user_id': user_id,
                    'username': user_data.get('username', 'Unknown'),
                    'roles': user_data.get('current_roles', []),
                    'message_count': message_count,
                    'content_count': content_count,
                    'all_message_counts': all_message_counts,
                    'channel_message_counts': channel_message_counts,
                    'all_content_counts': content_counts,
                    'channels': list(set([msg.get('channel_name', 'Unknown') for msg in user_data.get('messages', [])])),
                    'x_links': x_links_in_period
                })
                
        except Exception as e:
            logger.warning(f"Error processing user {user_id}: {e}")
            continue
    
    return filtered_results

def get_user_roles(member):
    """Returns user's roles with both names and IDs"""
    roles_data = []
    for role in member.roles:
        if role.name != '@everyone':
            roles_data.append({
                'name': role.name,
                'id': str(role.id)
            })
    return roles_data

@bot.event
async def on_ready():
    logger.info(f'Logged in as {bot.user}!')
    logger.info('Bot is ready and running!')

@bot.event
async def on_message(message):
    # Don't track bot's own messages
    if message.author.bot:
        return

    # Only track users with tracked roles
    try:
        member = await message.guild.fetch_member(message.author.id)
        user_role_ids = [role.id for role in member.roles]

        # Check if user has any tracked role
        has_tracked_role = any(role_id in TRACKED_ROLE_IDS for role_id in user_role_ids)

        # If no tracked roles configured, track everyone (for initial setup)
        if not TRACKED_ROLE_IDS:
            has_tracked_role = True

        if not has_tracked_role:
            # Process commands but don't track messages
            await bot.process_commands(message)
            return

    except discord.NotFound:
        logger.warning(f"Member {message.author.id} not found in guild")
        await bot.process_commands(message)
        return
    except discord.Forbidden:
        logger.warning(f"No permission to fetch member {message.author.id}")
        await bot.process_commands(message)
        return
    except Exception as e:
        logger.error(f"Error checking user roles for {message.author.id}: {e}")
        # If error occurs, still process commands
        await bot.process_commands(message)
        return

    # Load data
    data = load_data()

    # User ID
    user_id = str(message.author.id)

    # Check if this is a content channel and track Twitter/X links
    if CONTENT_CHANNEL_IDS and str(message.channel.id) in [str(ch_id) for ch_id in CONTENT_CHANNEL_IDS]:
        twitter_links = extract_twitter_links(message.content)
        if twitter_links:
            # Ensure user exists in data for content storage
            if user_id not in data:
                data[user_id] = {
                    'username': message.author.name,
                    'current_roles': [],
                    'messages': [],
                    'contents': []
                }

            # Build global set of already stored links to avoid duplicates (across all users)
            existing_links = set()
            try:
                for _, udata in data.items():
                    for entry in udata.get('contents', []):
                        for link in entry.get('twitter_links', []):
                            existing_links.add(link)
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"Error processing existing content links: {e}")

            # Keep only brand-new links
            new_links = [link for link in twitter_links if link not in existing_links]

            if new_links:
                # Append to user's contents in main data file
                user_contents = data[user_id].setdefault('contents', [])
                user_contents.append({
                    'timestamp': datetime.now().isoformat(),
                    'channel_id': str(message.channel.id),
                    'twitter_links': new_links,
                })

                # Save back to primary data file
                save_data(data)
                logger.info(f"Content tracked: {len(new_links)} X/Twitter links from {message.author.name}")

    # Get current user roles
    try:
        member = await message.guild.fetch_member(message.author.id)
        current_roles = get_user_roles(member)
    except discord.NotFound:
        logger.warning(f"Member {message.author.id} not found when getting roles")
        current_roles = []
    except discord.Forbidden:
        logger.warning(f"No permission to fetch member {message.author.id} for roles")
        current_roles = []
    except Exception as e:
        logger.error(f"Error fetching member roles for {message.author.id}: {e}")
        current_roles = []

    # Create user if not exists
    if user_id not in data:
        data[user_id] = {
            'username': message.author.name,
            'current_roles': current_roles,
            'messages': []
        }
    else:
        # Update current roles if user exists
        data[user_id]['current_roles'] = current_roles

    # Add message information (without content for privacy and performance)
    message_data = {
        'timestamp': datetime.now().isoformat(),
        'channel_id': str(message.channel.id),
        'channel_name': message.channel.name
    }

    data[user_id]['messages'].append(message_data)

    # Save data
    save_data(data)

    # Process commands
    await bot.process_commands(message)

@bot.event
async def on_member_update(before, after):
    """Update user roles when they change"""
    if before.roles != after.roles:
        # Load data
        data = load_data()

        user_id = str(after.id)

        # Update roles if user exists in our data
        if user_id in data:
            current_roles = get_user_roles(after)
            data[user_id]['current_roles'] = current_roles
            save_data(data)
            logger.info(f"Updated roles for user {after.name}: {current_roles}")

            # Log role changes to audit log
            try:
                async for entry in after.guild.audit_logs(limit=1, action=discord.AuditLogAction.member_role_update):
                    if entry.target.id == after.id:
                        logger.info(f"Audit Log - Role change by {entry.user.name}: {len(entry.before.roles)} -> {len(entry.after.roles)} roles")
                        break
            except discord.Forbidden:
                logger.warning("No permission to access audit logs")
            except Exception as e:
                logger.error(f"Audit log error: {e}")

@bot.event
async def on_user_update(before, after):
    """Update username when user changes their name"""
    if before.name != after.name:
        # Load data
        data = load_data()

        user_id = str(after.id)

        # Update username if user exists in our data
        if user_id in data:
            data[user_id]['username'] = after.name
            save_data(data)
            logger.info(f"Updated username for user {after.name}")

@bot.command(name='filter')
@commands.check(has_permission)
async def filter_users(ctx, *args):
    """
    Ultra-advanced filter for users with complex role criteria
    Usage: !filter [filters] [role_conditions] [time]
    Role Format: Use @rolename or 'roleid 123456789012345678' (with space)
    Examples:
    !filter message>1000 content>10 @Admin 7d    - Messages >1000, Content >10, Admin role, 7 days
    !filter message>500 @Admin and @Moderator 30d - Messages >500, Admin AND Moderator roles, 30 days
    !filter content>5 @VIP or @Premium 24h        - Content >5, VIP OR Premium roles, 24 hours
    !filter message>1000 @Admin nohave @Manager 7d - Messages >1000, Admin role but NOT Manager, 7 days
    !filter message>2000 roleid 123456789012345678 and roleid 987654321098765432 nohave roleid 555666777888999111 90d - Complex role ID filtering
    """

    # Clean arguments - remove invisible characters and empty strings
    cleaned_args = []
    for arg in args:
        # Remove invisible characters like word joiner (\u2060), zero-width space, etc.
        cleaned_arg = ''.join(char for char in arg if char.isprintable() and ord(char) >= 32)
        if cleaned_arg.strip():  # Only add non-empty args
            cleaned_args.append(cleaned_arg.strip())
    
    args = cleaned_args
    logger.debug(f"Cleaned args: {args}")
    
    # Parse arguments with advanced role logic
    filters = {}
    time_period = 'total'  # Changed default to total instead of 30d
    role_conditions = {
        'must_have': [],      # AND roles - user MUST have ALL of these
        'any_of': [],         # OR roles - user MUST have at least ONE of these
        'must_not_have': []   # NOHAVE roles - user MUST NOT have ANY of these
    }

    # Helper utilities for role parsing (supports spaces and quotes)
    def _find_role_by_name_exact(name: str):
        for r in ctx.guild.roles:
            if r.name.lower() == name.lower():
                return r
        return None

    def _parse_multiword_role(start_index: int):
        """Parse role tokens starting at an '@' token.
        Returns (role_id_str, extra_tokens_consumed).
        extra_tokens_consumed does NOT include the first token itself.
        """
        if start_index >= len(args):
            return None, 0

        token = args[start_index]
        if not token.startswith('@'):
            return None, 0

        # Remove initial '@'
        first_part = token[1:]
        name_parts = [first_part]
        j = start_index + 1

        # If quoted, keep consuming until closing quote
        if first_part.startswith('"'):
            name_parts[0] = name_parts[0][1:]
            while j < len(args):
                part = args[j]
                if part.endswith('"'):
                    name_parts.append(part[:-1])
                    j += 1
                    break
                else:
                    name_parts.append(part)
                    j += 1
        else:
            # Consume until a delimiter token (operator/other filters/channel/time) is reached
            # Also check if the next token starts with @ (another role)
            while j < len(args):
                peek = args[j]
                lower = peek.lower()
                if (
                    lower in ['and', 'or', 'nohave']
                    or peek.startswith('@')  # Another role mention
                    or peek.startswith('#')
                    or peek.startswith('<#')
                    or peek.startswith('<@&')  # Discord role mention
                    or lower in TIME_FILTERS
                    or ('>' in peek and not peek.startswith('@'))  # Filter but not role mention
                    or peek.startswith('top')
                    or lower == 'roleid'
                ):
                    break
                name_parts.append(peek)
                j += 1

        candidate_name = ' '.join([p for p in name_parts if p]).strip()
        logger.debug(f"Parsing multiword role: '{candidate_name}' from tokens {name_parts}")
        
        role_obj = _find_role_by_name_exact(candidate_name)
        if role_obj:
            logger.debug(f"Found role: {candidate_name} -> ID: {role_obj.id}")
            return str(role_obj.id), (j - start_index - 1)
        
        logger.debug(f"Role not found: '{candidate_name}'")
        return None, 0

    i = 0
    while i < len(args):
        arg = args[i]
        logger.debug(f"Processing argument {i}: '{arg}' (type: {type(arg)})")

        if arg in TIME_FILTERS:
            time_period = arg
        elif arg.startswith('#'):
            # Channel filter
            filters['channel'] = arg[1:]  # Remove # prefix
            logger.debug(f"Added channel filter: {filters['channel']}")
        elif arg.startswith('<#') and arg.endswith('>'):
            # Discord channel mention format <#channel_id>
            try:
                mention_id = arg[2:-1]
            except Exception:
                mention_id = ''
            if mention_id.isdigit():
                filters['channel_id'] = mention_id
            else:
                await ctx.send("❌ Invalid channel mention format!")
                return
        elif arg.startswith('<@&') and arg.endswith('>'):
            # Discord role mention format <@&role_id>
            try:
                role_id = arg[3:-1]  # Remove <@& and >
                logger.debug(f"Processing Discord role mention: {arg} -> role_id: {role_id}")
                if role_id.isdigit():
                    # Check if next argument is 'or' to determine where to add this role
                    if i + 1 < len(args) and args[i + 1].lower() == 'or':
                        role_conditions['any_of'].append(role_id)
                        logger.debug(f"Added Discord role {role_id} to any_of (because next is 'or')")
                    else:
                        role_conditions['must_have'].append(role_id)
                        logger.debug(f"Added Discord role {role_id} to must_have conditions")
                else:
                    await ctx.send("❌ Invalid role mention format!")
                    return
            except Exception as e:
                logger.error(f"Error parsing role mention {arg}: {e}")
                await ctx.send("❌ Error parsing role mention!")
                return
        elif '>' in arg:
            # Filter criteria like message>1000, content>10
            filter_type, value = arg.split('>', 1)
            logger.debug(f"Parsing filter: {filter_type}>{value}")
            try:
                filters[filter_type] = int(value)
                logger.debug(f"Successfully parsed filter: {filter_type}={int(value)}")
            except ValueError:
                logger.error(f"Invalid filter value - arg: '{arg}', filter_type: '{filter_type}', value: '{value}'")
                await ctx.send(f"❌ Invalid filter value: {value} (from {arg})")
                return
        elif arg.startswith('top'):
            # Top parameter
            try:
                limit = int(arg[3:])  # Remove 'top' prefix
                if limit <= 0 or limit > 50:
                    await ctx.send("❌ Top limit must be between 1-50!")
                    return
                filters['limit'] = limit
            except ValueError:
                await ctx.send("❌ Invalid top format! Use 'top10', 'top20', etc.")
                return
        elif arg.lower() in ['and', 'or', 'nohave']:
            # Role operators
            operator = arg.lower()
            logger.debug(f"Processing operator: {operator} at position {i}")

            # Get the next argument as role
            if i + 1 < len(args):
                next_arg = args[i + 1]
                role_id = None
                logger.debug(f"Next arg after {operator}: {next_arg}")

                if next_arg == 'roleid' and i + 2 < len(args):
                    # Format: roleid 123456789012345678 (with space)
                    role_id_str = args[i + 2]
                    logger.debug(f"Processing roleid format: {next_arg} {role_id_str}")
                    if role_id_str.isdigit() and len(role_id_str) >= 17 and len(role_id_str) <= 20:
                        role_id = role_id_str
                        logger.debug(f"Valid role ID found: {role_id}")
                        i += 1  # Skip the role ID argument as well
                    else:
                        await ctx.send(f"❌ Invalid role ID format: {role_id_str}. Use roleid 123456789012345678")
                        return
                elif next_arg.startswith('<@&') and next_arg.endswith('>'):
                    # Discord role mention format <@&role_id>
                    try:
                        role_id = next_arg[3:-1]  # Remove <@& and >
                        logger.debug(f"Processing Discord role mention after {operator}: {next_arg} -> role_id: {role_id}")
                        if role_id.isdigit():
                            logger.debug(f"Valid Discord role ID found: {role_id}")
                        else:
                            await ctx.send("❌ Invalid Discord role mention format!")
                            return
                    except Exception as e:
                        logger.error(f"Error parsing Discord role mention {next_arg}: {e}")
                        await ctx.send("❌ Error parsing Discord role mention!")
                        return
                elif next_arg.startswith('@'):
                    # Role mention like @Admin or @"Waiting Room" or @Waiting Room
                    parsed_role_id, extra_tokens = _parse_multiword_role(i + 1)
                    if parsed_role_id:
                        role_id = parsed_role_id
                        i += extra_tokens  # consume additional tokens used by multiword role
                        logger.debug(f"Parsed multiword role -> id: {role_id}")
                    else:
                        # Fallback to single-token name first
                        role_name = next_arg[1:]
                        role_obj = _find_role_by_name_exact(role_name)
                        if role_obj:
                            role_id = str(role_obj.id)
                            logger.debug(f"Found role by name: {role_name} -> {role_id}")
                        else:
                            # Try multiword parsing manually
                            multiword_parts = [role_name]
                            j = i + 2  # Start from the token after next_arg
                            while j < len(args):
                                token = args[j]
                                if (token.lower() in ['and', 'or', 'nohave'] or 
                                    token.startswith('@') or 
                                    token.startswith('#') or 
                                    token.startswith('<') or
                                    token.lower() in TIME_FILTERS or
                                    '>' in token or
                                    token.startswith('top') or
                                    token.lower() == 'roleid'):
                                    break
                                multiword_parts.append(token)
                                j += 1
                            
                            full_role_name = ' '.join(multiword_parts)
                            role_obj = _find_role_by_name_exact(full_role_name)
                            if role_obj:
                                role_id = str(role_obj.id)
                                i += (j - i - 2)  # consume the extra tokens
                                logger.debug(f"Found multiword role: {full_role_name} -> {role_id}")
                            else:
                                await ctx.send(f"❌ Role not found: @{full_role_name}")
                                return
                else:
                    await ctx.send(f"❌ Invalid role format: {next_arg}. Use 'roleid 123456789012345678', @rolename, or <@&role_id>")
                    return

                # Add role to appropriate condition based on operator
                if role_id:  # Make sure we have a valid role_id
                    if operator == 'and':
                        role_conditions['must_have'].append(role_id)
                        logger.debug(f"Added {role_id} to must_have")
                    elif operator == 'or':
                        role_conditions['any_of'].append(role_id)
                        logger.debug(f"Added {role_id} to any_of")
                    elif operator == 'nohave':
                        role_conditions['must_not_have'].append(role_id)
                        logger.debug(f"Added {role_id} to must_not_have")
                else:
                    await ctx.send(f"❌ Failed to parse role after {operator} operator")
                    return

                i += 1  # Skip next argument
            else:
                await ctx.send("❌ Missing role after operator")
                return
        elif arg == 'roleid' and i + 1 < len(args):
            # Single role ID with space: roleid 123456789012345678
            role_id_str = args[i + 1]
            logger.debug(f"Processing single roleid: {role_id_str}")
            if role_id_str.isdigit() and len(role_id_str) >= 17 and len(role_id_str) <= 20:
                # Check if the argument after role_id is 'or' - if so, this should go to any_of
                if i + 2 < len(args) and args[i + 2].lower() == 'or':
                    role_conditions['any_of'].append(role_id_str)
                    logger.debug(f"Added {role_id_str} to any_of (because next is 'or')")
                else:
                    role_conditions['must_have'].append(role_id_str)
                    logger.debug(f"Added {role_id_str} to must_have")
                i += 1  # Skip the role ID argument
            else:
                await ctx.send(f"❌ Invalid role ID format: {role_id_str}. Use roleid 123456789012345678")
                return
        elif arg.startswith('@'):
            # Single role mention like @Admin or @Waiting Room (with spaces) or @"Role Name"
            parsed_role_id, extra_tokens = _parse_multiword_role(i)
            if parsed_role_id:
                # Check if the next argument (after consuming extra tokens) is 'or'
                next_arg_index = i + extra_tokens + 1
                if next_arg_index < len(args) and args[next_arg_index].lower() == 'or':
                    role_conditions['any_of'].append(parsed_role_id)
                    logger.debug(f"Added role {parsed_role_id} to any_of (because next is 'or')")
                else:
                    role_conditions['must_have'].append(parsed_role_id)
                    logger.debug(f"Added role {parsed_role_id} to must_have")
                i += extra_tokens  # we will still increment by 1 at loop end
            else:
                # Fallback single-token behavior - try just the first token after @
                role_name = arg[1:]
                role_obj = _find_role_by_name_exact(role_name)
                if role_obj:
                    if i + 1 < len(args) and args[i + 1].lower() == 'or':
                        role_conditions['any_of'].append(str(role_obj.id))
                        logger.debug(f"Added @{role_name} ({role_obj.id}) to any_of (because next is 'or')")
                    else:
                        role_conditions['must_have'].append(str(role_obj.id))
                        logger.debug(f"Added @{role_name} ({role_obj.id}) to must_have")
                else:
                    # If single token failed, try to construct multiword role name manually
                    # by consuming tokens until we find a delimiter or role
                    multiword_parts = [role_name]
                    j = i + 1
                    while j < len(args):
                        next_token = args[j]
                        if (next_token.lower() in ['and', 'or', 'nohave'] or 
                            next_token.startswith('@') or 
                            next_token.startswith('#') or 
                            next_token.startswith('<') or
                            next_token.lower() in TIME_FILTERS or
                            '>' in next_token or
                            next_token.startswith('top') or
                            next_token.lower() == 'roleid'):
                            break
                        multiword_parts.append(next_token)
                        j += 1
                    
                    full_role_name = ' '.join(multiword_parts)
                    role_obj = _find_role_by_name_exact(full_role_name)
                    if role_obj:
                        extra_consumed = j - i - 1
                        if j < len(args) and args[j].lower() == 'or':
                            role_conditions['any_of'].append(str(role_obj.id))
                            logger.debug(f"Added multiword @{full_role_name} ({role_obj.id}) to any_of (because next is 'or')")
                        else:
                            role_conditions['must_have'].append(str(role_obj.id))
                            logger.debug(f"Added multiword @{full_role_name} ({role_obj.id}) to must_have")
                        i += extra_consumed
                    else:
                        await ctx.send(f"❌ Role not found: @{full_role_name}")
                        return
        else:
            await ctx.send(f"❌ Invalid argument: {arg}. Use role ID, @rolename, time period, or filters")
            return

        i += 1

    logger.debug(f"Parsed filters: {filters}, time: {time_period}, role_conditions: {role_conditions}")
    logger.info(f"Filter command - Role conditions: must_have={role_conditions['must_have']}, any_of={role_conditions['any_of']}, must_not_have={role_conditions['must_not_have']}")

    # Validate time filter
    if time_period not in TIME_FILTERS:
        await ctx.send(f"Invalid time filter! Available filters: {', '.join(TIME_FILTERS.keys())}")
        return

    # Calculate time limit (0 for total means no limit)
    time_limit = datetime.now() - timedelta(seconds=TIME_FILTERS[time_period]) if TIME_FILTERS[time_period] > 0 else datetime.min
    logger.debug(f"Filtering: filters={filters}, time={time_period}, role_conditions={role_conditions}")

    # Calculate statistics for all time periods
    now = datetime.now()
    all_time_periods = {
        '24h': now - timedelta(hours=24),
        '7d': now - timedelta(days=7),
        '30d': now - timedelta(days=30),
        '90d': now - timedelta(days=90),
        '180d': now - timedelta(days=180),
        '360d': now - timedelta(days=360),
        'total': datetime.min
    }
    
    # Add time period to filters for chunk processing
    filters['time_period'] = time_period
    
    # Start optimized processing
    logger.info(f"Starting optimized filtering for 1M+ users - using streaming and parallel processing")
    start_time = datetime.now()

    # Use optimized streaming processing for 1M+ users
    try:
        # Try to use standard loading for smaller datasets first
        file_size = DATA_FILE.stat().st_size if DATA_FILE.exists() else 0
        file_size_mb = file_size / (1024 * 1024)
        
        if file_size_mb > 100:  # If file is larger than 100MB, use streaming
            logger.info(f"Large dataset detected ({file_size_mb:.1f}MB) - using streaming processing")
            filtered_results = await process_large_dataset_streaming(ctx, filters, all_time_periods, role_conditions, time_limit)
        else:
            logger.info(f"Small dataset ({file_size_mb:.1f}MB) - using standard processing")
            # Load data normally for smaller datasets
            data = load_data()
            filtered_results = process_standard_dataset(data, filters, all_time_periods, role_conditions, time_limit, ctx)
            
    except Exception as e:
        logger.error(f"Error during optimized processing: {e}")
        # Fallback to standard processing
        logger.info("Falling back to standard processing")
        data = load_data()
        filtered_results = process_standard_dataset(data, filters, all_time_periods, role_conditions, time_limit, ctx)

    # Log performance metrics
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    logger.info(f"Filter processing completed in {processing_time:.2f} seconds - {len(filtered_results)} results found")
    
    if not filtered_results:
        # If no filters were applied and no results, show message about criteria
        if filters or role_conditions['must_have'] or role_conditions['any_of'] or role_conditions['must_not_have']:
            await ctx.send(f"No users found matching the specified criteria! (Processed in {processing_time:.1f}s)")
        else:
            await ctx.send(f"No users found in the database! (Processed in {processing_time:.1f}s)")
        return

    # Sort by message count (highest first) and apply limit if specified
    filtered_results.sort(key=lambda x: x['message_count'], reverse=True)

    limit = filters.get('limit', None)
    if limit:
        filtered_results = filtered_results[:limit]
        display_results = filtered_results
        result_text = f"Top {limit} users"
    else:
        display_results = filtered_results
        result_text = f"{len(filtered_results)} users"

    # Create filter description
    filter_desc = []
    if time_period:
        filter_desc.append(f"Time: {time_period.upper()}")

    # Add filter criteria to description
    filter_criteria = []
    if 'message' in filters:
        filter_criteria.append(f"Messages >{filters['message']}")
    if 'content' in filters:
        filter_criteria.append(f"Content >{filters['content']}")

    # Add role conditions to description
    role_parts = []
    if role_conditions['must_have']:
        role_names = []
        for role_id in role_conditions['must_have']:
            role_obj = ctx.guild.get_role(int(role_id))
            role_names.append(role_obj.name if role_obj else role_id)
        role_parts.append(f"MUST: {', '.join(role_names)}")

    if role_conditions['any_of']:
        role_names = []
        for role_id in role_conditions['any_of']:
            role_obj = ctx.guild.get_role(int(role_id))
            role_names.append(role_obj.name if role_obj else role_id)
        role_parts.append(f"ANY: {', '.join(role_names)}")

    if role_conditions['must_not_have']:
        role_names = []
        for role_id in role_conditions['must_not_have']:
            role_obj = ctx.guild.get_role(int(role_id))
            role_names.append(role_obj.name if role_obj else role_id)
        role_parts.append(f"NOT: {', '.join(role_names)}")

    if role_parts:
        filter_criteria.append(f"Roles ({' & '.join(role_parts)})")

    # Add channel description if filtered
    if 'channel' in filters:
        filter_criteria.append(f"Channel: #{filters['channel']}")
    if 'channel_id' in filters:
        filter_criteria.append(f"ChannelID: {filters['channel_id']}")

    if filter_criteria:
        filter_desc.extend(filter_criteria)

    filter_description = " | ".join(filter_desc) if filter_desc else "All users"

    # Create results message
    results_text = f"📊 **Advanced Filter Results: {filter_description}**\n"
    results_text += f"📈 **{result_text} found**\n\n"

    for i, result in enumerate(display_results, 1):
        rank_emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
        results_text += f"{rank_emoji} **{result['username']}**\n"
        results_text += f"   📝 Messages: {result['message_count']}\n"
        results_text += f"   📊 Content: {result['content_count']}\n"
        # Handle both old and new role format
        if result['roles'] and isinstance(result['roles'][0], dict):
            role_names = [role['name'] for role in result['roles']]
        else:
            role_names = result['roles']
        results_text += f"   👑 Roles: {', '.join(role_names)}\n\n"

    # Create Excel file
    filename = f"advanced_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    create_advanced_excel_report(filtered_results, filters, time_period, role_conditions, filename, data)

    # Send only the Excel file (no text output)
    await ctx.send(file=discord.File(filename))

    # Delete temporary file
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not delete temporary file {filename}: {e}")

def create_excel_report(results, filename):
    """Create Excel report"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Discord Report"

    # Add headers
    headers = ['Username', 'User ID', 'Roles', 'Message Count', 'Channels']
    for col_num, header in enumerate(headers, 1):
        ws.cell(row=1, column=col_num, value=header)

    # Add data
    for row_num, result in enumerate(results, 2):
        ws.cell(row=row_num, column=1, value=result['username'])
        ws.cell(row=row_num, column=2, value=result['user_id'])
        # Handle both old and new role format
        if result['roles'] and isinstance(result['roles'][0], dict):
            role_names = [role['name'] for role in result['roles']]
        else:
            role_names = result['roles']
        ws.cell(row=row_num, column=3, value=', '.join(role_names))
        ws.cell(row=row_num, column=4, value=result['message_count'])
        ws.cell(row=row_num, column=5, value=', '.join(result['channels']))

    # Adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = min(max_length + 2, 50)  # Maximum 50 characters
        ws.column_dimensions[column_letter].width = adjusted_width

    # Save
    wb.save(filename)

def create_stats_excel(time_stats, filename):
    """Create Excel report for statistics"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Statistics Report"

    # Load data to get actual totals
    data = load_data()

    # Add headers
    headers = ['Time Period', 'Messages', 'Active Users']
    for col_num, header in enumerate(headers, 1):
        ws.cell(row=1, column=col_num, value=header)

    # Add data
    row_num = 2
    for period_name, stats in time_stats.items():
        ws.cell(row=row_num, column=1, value=period_name.upper())
        ws.cell(row=row_num, column=2, value=stats['messages'])
        ws.cell(row=row_num, column=3, value=stats['users'])
        row_num += 1

    # Add summary - Use actual total counts from all data
    total_users = len(data)
    total_messages = sum(len(user_data['messages']) for user_data in data.values())
    # Content count is not stored in user data, calculate from content_data
    content_data = load_content_data()
    total_content = len(content_data)
    ws.cell(row=row_num, column=1, value="TOTAL")
    ws.cell(row=row_num, column=2, value=total_messages)
    ws.cell(row=row_num, column=3, value=total_users)

    # Adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = min(max_length + 2, 30)
        ws.column_dimensions[column_letter].width = adjusted_width

    # Save
    wb.save(filename)





def create_advanced_excel_report(results, filters, time_period, role_conditions, filename, data=None):
    """Create Excel report for advanced filtering with all time periods"""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Advanced Filter Report"

    # Add headers for all time periods (channel-specific + general stats)
    channel_filter_active = any(key in filters for key in ['channel', 'channel_id'])
    if channel_filter_active:
        # Get channel name - if channel_id is provided, find the name from data
        channel_name = filters.get('channel')
        if not channel_name and 'channel_id' in filters and data:
            # Look for channel name in the data by channel_id
            target_id = str(filters['channel_id'])
            for user_data in data.values():
                for msg in user_data.get('messages', []):
                    if str(msg.get('channel_id')) == target_id:
                        channel_name = msg.get('channel_name', target_id)
                        break
                if channel_name:
                    break
            if not channel_name:
                channel_name = target_id
        
        headers = ['Username', 'User ID', 'Roles',
                   f'Ch-24H ({channel_name})', f'Ch-7D ({channel_name})', f'Ch-30D ({channel_name})', 
                   f'Ch-90D ({channel_name})', f'Ch-180D ({channel_name})', f'Ch-360D ({channel_name})', f'Ch-Total ({channel_name})',
                   '24H Messages', '7D Messages', '30D Messages', '90D Messages', '180D Messages', '360D Messages', 'Total Messages',
                   '24H Content', '7D Content', '30D Content', '90D Content', '180D Content', '360D Content', 'Total Content',
                   'X Links', 'Channels']
    else:
        headers = ['Username', 'User ID', 'Roles',
                   '24H Messages', '7D Messages', '30D Messages', '90D Messages', '180D Messages', '360D Messages', 'Total Messages',
                   '24H Content', '7D Content', '30D Content', '90D Content', '180D Content', '360D Content', 'Total Content',
                   'X Links', 'Channels']

    for col_num, header in enumerate(headers, 1):
        ws.cell(row=1, column=col_num, value=header)

    # Add data
    for row_num, result in enumerate(results, 2):
        col_num = 1
        ws.cell(row=row_num, column=col_num, value=result['username'])
        col_num += 1
        ws.cell(row=row_num, column=col_num, value=result['user_id'])
        col_num += 1
        # Handle both old and new role format
        if result['roles'] and isinstance(result['roles'][0], dict):
            role_names = [role['name'] for role in result['roles']]
        else:
            role_names = result['roles']
        ws.cell(row=row_num, column=col_num, value=', '.join(role_names))
        col_num += 1

        # Channel-specific message counts first (if channel filter is active)
        if channel_filter_active:
            for period in ['24h', '7d', '30d', '90d', '180d', '360d', 'total']:
                ws.cell(row=row_num, column=col_num, value=result['channel_message_counts'][period])
                col_num += 1

        # All message counts
        for period in ['24h', '7d', '30d', '90d', '180d', '360d', 'total']:
            ws.cell(row=row_num, column=col_num, value=result['all_message_counts'][period])
            col_num += 1

        # Content counts for all periods
        for period in ['24h', '7d', '30d', '90d', '180d', '360d', 'total']:
            ws.cell(row=row_num, column=col_num, value=result['all_content_counts'][period])
            col_num += 1

        # X Links for selected time period - write newest link as clickable and rest below
        x_links_list = result.get('x_links', [])
        display_text = '\n'.join(x_links_list)
        cell = ws.cell(row=row_num, column=col_num, value=display_text)
        if x_links_list:
            try:
                # Set hyperlink to the newest (first) link
                cell.hyperlink = x_links_list[0]
                # Apply hyperlink styling
                cell.font = Font(color="0000FF", underline="single")
            except Exception:
                pass
        col_num += 1

        # Channels
        ws.cell(row=row_num, column=col_num, value=', '.join(result['channels']))

    # Add filter summary
    summary_row = len(results) + 4
    ws.cell(row=summary_row, column=1, value="FILTER SUMMARY")
    ws.cell(row=summary_row, column=2, value=f"Time Period: {time_period.upper()}")
    
    # Create role description from role_conditions
    role_desc_parts = []
    if role_conditions['must_have']:
        role_desc_parts.append(f"MUST: {', '.join(role_conditions['must_have'])}")
    if role_conditions['any_of']:
        role_desc_parts.append(f"ANY: {', '.join(role_conditions['any_of'])}")
    if role_conditions['must_not_have']:
        role_desc_parts.append(f"NOT: {', '.join(role_conditions['must_not_have'])}")
    
    role_description = ' & '.join(role_desc_parts) if role_desc_parts else 'All'
    
    ws.cell(row=summary_row, column=3, value=f"Roles: {role_description}")
    ws.cell(row=summary_row, column=4, value=f"Message Filter: >{filters.get('message', 'None')}")
    ws.cell(row=summary_row, column=5, value=f"Content Filter: >{filters.get('content', 'None')}")
    ws.cell(row=summary_row, column=6, value=f"Total Results: {len(results)}")

    # Adjust column widths
    for column in ws.columns:
        max_length = 0
        column_letter = get_column_letter(column[0].column)
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = min(max_length + 2, 25)
        ws.column_dimensions[column_letter].width = adjusted_width

    # Save
    wb.save(filename)

@bot.command(name='trackhelp')
@commands.check(has_permission)
async def trackhelp(ctx):
    """Show all available commands for the tracking bot in plain text"""

    help_text = """
📊 **Discord Message Tracking Bot - All Commands**
⚠️ **IMPORTANT: Bot only tracks users with tracked roles!**

🔍 **!filter [time] [role] [channel]**
Description: Filter users and create Excel report with message statistics
Time Options: 24h, 7d, 30d, 90d, 180d, 360d

Examples:
• `!filter message>1000 content>10 @Admin` - Messages >1000, Content >10, Admin role (total/all time)
• `!filter message>1000 content>10 @Admin 7d` - Messages >1000, Content >10, Admin role, 7 days
• `!filter message>500 @Admin and @Moderator 30d` - Messages >500, Admin AND Moderator roles, 30 days
• `!filter content>5 @VIP or @Premium 24h` - Content >5, VIP OR Premium roles, 24 hours
• `!filter message>1000 @Admin nohave @Manager 7d` - Messages >1000, Admin role but NOT Manager role, 7 days
• `!filter message>2000 @Admin and @Helper nohave @Muted 90d` - Admin & Helper roles but not Muted, 90 days
• `!filter message>1000 content>10 roleid 123456789012345678 7d` - Using role ID with space
• `!filter message>500 @Admin 30d top10` - Top 10 Admin users with most messages (30 days)
• `!filter content>20 @VIP 7d top50` - Top 50 VIP users with most content (7 days)
• `!filter message>1000 @Admin and @Moderator 90d top100` - Top 100 users with both roles (90 days)
• `!filter message>1000 content>5 #general 7d` - Messages >1000, Content >5, in #general channel, 7 days
• `!filter content>10 @VIP or @Premium #announcements 30d` - Content >10, VIP OR Premium, in #announcements, 30 days
• `!filter message>500 @Admin nohave @Manager #help-desk 24h` - Admin but not Manager, in #help-desk, 24 hours
• `!filter message>1000 roleid 123456789012345678 and roleid 987654321098765432 nohave roleid 555666777888999111 7d` - Complex role ID filtering

📈 **!stats**
Description: Show general server statistics
Shows: Total users, total messages, messages from last 24 hours
Usage: `!stats`

❓ **!help**
Description: Show basic help message
Usage: `!help`

🎯 **!trackhelp**
Description: Show this detailed command list with examples
Usage: `!trackhelp`

👑 **!adduser [user_id]**
Description: Add authorized user (owner only)
Usage: `!adduser 123456789012345678`

👑 **!removeuser [user_id]**
Description: Remove authorized user (owner only)
Usage: `!removeuser 123456789012345678`

👑 **!addrole [role_id]**
Description: Add authorized role (owner only)
Usage: `!addrole 987654321098765432`

👑 **!removerole [role_id]**
Description: Remove authorized role (owner only)
Usage: `!removerole 987654321098765432`

📊 **!addtrackedrole [role_id]**
Description: Add role for message tracking (owner only)
Usage: `!addtrackedrole 123456789012345678`

📊 **!removetrackedrole [role_id]**
Description: Remove tracked role (owner only)
Usage: `!removetrackedrole 123456789012345678`

🔒 **!listauth**
Description: List all authorized users and roles
Usage: `!listauth`

📺 **!addcontentchannel [channel_id]**
Description: Add content channel for Twitter/X link tracking (bot owner only)
Usage: `!addcontentchannel 123456789012345678`

📺 **!removecontentchannel [channel_id]**
Description: Remove content channel (bot owner only)
Usage: `!removecontentchannel 123456789012345678`

📺 **!listcontentchannels**
Description: List all content channels
Usage: `!listcontentchannels`



ℹ️ **Bot Features:**
• **Role-Based Tracking: Only tracks users with tracked roles!**
• **Content Tracking: Automatically tracks Twitter/X links in multiple content channels**
• Real-time Role Tracking: Automatically updates user roles in JSON
• Audit Log Integration: Logs role changes with responsible user
• Excel Export: Generate detailed reports with user statistics
• Multi-filter Support: Filter by time, role, and channel
• Content Analysis: Track Twitter/X links and content statistics
• Persistent Storage: All data saved to message_tracking.json
• Automatic Updates: JSON updates when roles change
• Permission System: Only authorized users can use commands
• Rate Limiting: 5-second cooldown to prevent abuse
• Dynamic Authorization: Add/remove users/roles via commands
• Security Validation: Input validation for IDs
• Silent Protection: Unauthorized commands are deleted silently

⚙️ **Permission & Tracking Management:**
• Use `!adduser <user_id>` to add authorized users (owner only)
• Use `!removeuser <user_id>` to remove authorized users (owner only)
• Use `!addrole <role_id>` to add authorized roles (owner only)
• Use `!removerole <role_id>` to remove authorized roles (owner only)
• Use `!addtrackedrole <role_id>` to add roles for message tracking (owner only)
• Use `!removetrackedrole <role_id>` to remove tracked roles (owner only)
• Use `!listauth` to view all authorized users and roles
• Unauthorized users' commands are silently deleted
• Rate limiting: 5-second cooldown between commands

To get IDs: Enable Developer Mode in Discord settings, right-click user/role, and select "Copy ID"

Bot automatically tracks all messages and role changes in real-time.
"""

    # Split message if too long (Discord 2000 char limit)
    if len(help_text) > 2000:
        # Split into chunks
        chunks = []
        current_chunk = ""

        for line in help_text.split('\n'):
            if len(current_chunk + line + '\n') > 1900:  # Leave some buffer
                chunks.append(current_chunk)
                current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

        if current_chunk:
            chunks.append(current_chunk)

        # Send chunks
        for chunk in chunks:
            await ctx.send(chunk)
    else:
        await ctx.send(help_text)

@bot.command(name='stats')
@commands.check(has_permission)
async def stats(ctx):
    """Show general statistics for all time periods"""
    data = load_data()

    total_users = len(data)
    total_messages = sum(len(user_data['messages']) for user_data in data.values())

    embed = discord.Embed(
        title="📊 Message Statistics - All Time Periods",
        color=0x00ff00
    )

    # Overall statistics - Use actual total counts, not 360d approximation
    embed.add_field(name="👥 Total Users", value=str(total_users), inline=True)
    embed.add_field(name="💬 Total Messages", value=str(total_messages), inline=True)
    embed.add_field(name="📈 Average per User", value=f"{total_messages/total_users:.1f}" if total_users > 0 else "0", inline=True)

    # Calculate statistics for each time period
    now = datetime.now()
    time_periods = {
        '24h': timedelta(hours=24),
        '7d': timedelta(days=7),
        '30d': timedelta(days=30),
        '90d': timedelta(days=90),
        '180d': timedelta(days=180),
        '360d': timedelta(days=360)
    }

    # Time-based statistics
    time_stats = {}
    for period_name, time_delta in time_periods.items():
        period_start = now - time_delta
        period_messages = 0
        period_users = set()

        for user_id, user_data in data.items():
            user_messages = 0
            for msg in user_data['messages']:
                msg_time = datetime.fromisoformat(msg['timestamp'])
                if msg_time > period_start:
                    user_messages += 1

            if user_messages > 0:
                period_messages += user_messages
                period_users.add(user_id)

        time_stats[period_name] = {
            'messages': period_messages,
            'users': len(period_users)
        }

    # Display time-based statistics in fields
    for period_name, stats in time_stats.items():
        embed.add_field(
            name=f"📅 {period_name.upper()}",
            value=f"Messages: {stats['messages']}\nUsers: {stats['users']}",
            inline=True
        )

    # Most active time period
    if time_stats:
        most_active = max(time_stats.items(), key=lambda x: x[1]['messages'])
        embed.add_field(
            name="🏆 Most Active Period",
            value=f"{most_active[0].upper()}\n({most_active[1]['messages']} messages)",
            inline=False
        )

    embed.set_footer(text="Real-time statistics updated automatically")

    # Create Excel file for statistics
    filename = f"stats_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    create_stats_excel(time_stats, filename)

    # Send embed and Excel file publicly
    await ctx.send(embed=embed)
    await ctx.send("📊 **Detailed Statistics Excel Report:**", file=discord.File(filename))

    # Delete temporary file
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not delete temporary file {filename}: {e}")

# Permission management commands - Bot owner only
@bot.command(name='adduser')
@commands.check(has_owner_permission)
async def add_authorized_user(ctx, user_id: str):
    """Add authorized user (bot owner only)"""
    if not is_valid_user_id(user_id):
        await ctx.send("❌ Invalid user ID!")
        return

    user_id_int = int(user_id)

    if user_id_int in AUTHORIZED_USERS:
        await ctx.send("⚠️ This user is already authorized!")
        return

    if len(AUTHORIZED_USERS) >= MAX_AUTHORIZED_USERS:
        await ctx.send(f"❌ Maximum authorized users limit reached ({MAX_AUTHORIZED_USERS})!")
        return

    AUTHORIZED_USERS.append(user_id_int)
    await save_authorized_data()
    await ctx.send(f"✅ User {user_id} added to authorized list!")

@bot.command(name='removeuser')
@commands.check(has_owner_permission)
async def remove_authorized_user(ctx, user_id: str):
    """Remove authorized user (bot owner only)"""
    if not is_valid_user_id(user_id):
        await ctx.send("❌ Invalid user ID!")
        return

    user_id_int = int(user_id)

    if user_id_int not in AUTHORIZED_USERS:
        await ctx.send("⚠️ This user is not authorized!")
        return

    AUTHORIZED_USERS.remove(user_id_int)
    await save_authorized_data()
    await ctx.send(f"✅ User {user_id} removed from authorized list!")

@bot.command(name='addrole')
@commands.check(has_owner_permission)
async def add_authorized_role(ctx, role_id: str):
    """Add authorized role (bot owner only)"""
    if not is_valid_role_id(role_id):
        await ctx.send("❌ Invalid role ID!")
        return

    role_id_int = int(role_id)

    if role_id_int in AUTHORIZED_ROLE_IDS:
        await ctx.send("⚠️ This role is already authorized!")
        return

    if len(AUTHORIZED_ROLE_IDS) >= MAX_AUTHORIZED_ROLES:
        await ctx.send(f"❌ Maximum authorized roles limit reached ({MAX_AUTHORIZED_ROLES})!")
        return

    AUTHORIZED_ROLE_IDS.append(role_id_int)
    await save_authorized_data()
    await ctx.send(f"✅ Role {role_id} added to authorized list!")

@bot.command(name='removerole')
@commands.check(has_owner_permission)
async def remove_authorized_role(ctx, role_id: str):
    """Remove authorized role (bot owner only)"""
    if not is_valid_role_id(role_id):
        await ctx.send("❌ Invalid role ID!")
        return

    role_id_int = int(role_id)

    if role_id_int not in AUTHORIZED_ROLE_IDS:
        await ctx.send("⚠️ This role is not authorized!")
        return

    AUTHORIZED_ROLE_IDS.remove(role_id_int)
    await save_authorized_data()
    await ctx.send(f"✅ Role {role_id} removed from authorized list!")

@bot.command(name='addtrackedrole')
@commands.check(has_owner_permission)
async def add_tracked_role(ctx, role_id: str):
    """Add tracked role for message tracking (bot owner only)"""
    if not is_valid_role_id(role_id):
        await ctx.send("❌ Invalid role ID!")
        return

    role_id_int = int(role_id)

    if role_id_int in TRACKED_ROLE_IDS:
        await ctx.send("⚠️ This role is already being tracked!")
        return

    if len(TRACKED_ROLE_IDS) >= MAX_AUTHORIZED_ROLES:
        await ctx.send(f"❌ Maximum tracked roles limit reached ({MAX_AUTHORIZED_ROLES})!")
        return

    TRACKED_ROLE_IDS.append(role_id_int)
    await save_authorized_data()
    await ctx.send(f"✅ Role {role_id} added to tracked roles list!")

@bot.command(name='removetrackedrole')
@commands.check(has_owner_permission)
async def remove_tracked_role(ctx, role_id: str):
    """Remove tracked role (bot owner only)"""
    if not is_valid_role_id(role_id):
        await ctx.send("❌ Invalid role ID!")
        return

    role_id_int = int(role_id)

    if role_id_int not in TRACKED_ROLE_IDS:
        await ctx.send("⚠️ This role is not being tracked!")
        return

    TRACKED_ROLE_IDS.remove(role_id_int)
    await save_authorized_data()
    await ctx.send(f"✅ Role {role_id} removed from tracked roles list!")

@bot.command(name='addcontentchannel')
@commands.check(has_permission)
async def add_content_channel(ctx, channel_id: str):
    """Add content channel for Twitter/X link tracking (bot owner only)"""
    if not is_valid_channel_id(channel_id):
        await ctx.send("❌ Invalid channel ID!")
        return

    channel_id_int = int(channel_id)

    if channel_id_int in CONTENT_CHANNEL_IDS:
        await ctx.send("⚠️ This channel is already a content channel!")
        return

    CONTENT_CHANNEL_IDS.append(channel_id_int)
    save_config()

    # Verify channel exists
    try:
        channel = await bot.fetch_channel(channel_id_int)
        await ctx.send(f"✅ Content channel added: {channel.name} ({channel_id})")
    except discord.NotFound:
        await ctx.send(f"✅ Content channel ID added: {channel_id} (Channel not found)")
    except discord.Forbidden:
        await ctx.send(f"✅ Content channel ID added: {channel_id} (No access to channel)")
    except Exception as e:
        logger.warning(f"Error verifying channel {channel_id}: {e}")
        await ctx.send(f"✅ Content channel ID added: {channel_id} (Channel may not be accessible)")

@bot.command(name='removecontentchannel')
@commands.check(has_permission)
async def remove_content_channel(ctx, channel_id: str):
    """Remove content channel (bot owner only)"""
    if not is_valid_channel_id(channel_id):
        await ctx.send("❌ Invalid channel ID!")
        return

    channel_id_int = int(channel_id)

    if channel_id_int not in CONTENT_CHANNEL_IDS:
        await ctx.send("⚠️ This channel is not a content channel!")
        return

    CONTENT_CHANNEL_IDS.remove(channel_id_int)
    save_config()
    await ctx.send(f"✅ Content channel removed: {channel_id}")

@bot.command(name='listcontentchannels')
@commands.check(has_permission)
async def list_content_channels(ctx):
    """List all content channels"""
    if CONTENT_CHANNEL_IDS:
        channel_list = []
        for channel_id in CONTENT_CHANNEL_IDS:
            try:
                channel = await bot.fetch_channel(channel_id)
                channel_list.append(f"{channel.name} ({channel_id})")
            except discord.NotFound:
                channel_list.append(f"Unknown Channel ({channel_id})")
            except Exception as e:
                logger.warning(f"Error fetching channel {channel_id}: {e}")
                channel_list.append(f"Unknown Channel ({channel_id})")

        embed = discord.Embed(
            title="📺 Content Channels",
            description='\n'.join(channel_list),
            color=0x00ff00
        )
        await ctx.send(embed=embed)
    else:
        await ctx.send("📊 No content channels set!")





@bot.command(name='listauth')
@commands.check(has_permission)
async def list_authorized(ctx):
    """List authorized users and roles"""
    embed = discord.Embed(
        title="🔒 Authorized Users and Roles",
        color=0x00ff00
    )

    # Authorized users
    if AUTHORIZED_USERS:
        user_list = []
        for user_id in AUTHORIZED_USERS:
            try:
                user = await bot.fetch_user(user_id)
                user_list.append(f"{user.name} ({user_id})")
            except discord.NotFound:
                user_list.append(f"Unknown User ({user_id})")
            except Exception as e:
                logger.warning(f"Error fetching user {user_id}: {e}")
                user_list.append(f"Unknown User ({user_id})")
        embed.add_field(name="👥 Authorized Users", value='\n'.join(user_list), inline=False)
    else:
        embed.add_field(name="👥 Authorized Users", value="No authorized users", inline=False)

    # Authorized roles
    if AUTHORIZED_ROLE_IDS:
        role_list = []
        for role_id in AUTHORIZED_ROLE_IDS:
            try:
                role = ctx.guild.get_role(role_id)
                if role:
                    role_list.append(f"{role.name} ({role_id})")
                else:
                    role_list.append(f"Unknown Role ({role_id})")
            except Exception as e:
                logger.warning(f"Error fetching role {role_id}: {e}")
                role_list.append(f"Unknown Role ({role_id})")
        embed.add_field(name="👑 Authorized Roles", value='\n'.join(role_list), inline=False)
    else:
        embed.add_field(name="👑 Authorized Roles", value="No authorized roles", inline=False)

    # Tracked roles
    if TRACKED_ROLE_IDS:
        tracked_role_list = []
        for role_id in TRACKED_ROLE_IDS:
            try:
                role = ctx.guild.get_role(role_id)
                if role:
                    tracked_role_list.append(f"{role.name} ({role_id})")
                else:
                    tracked_role_list.append(f"Unknown Role ({role_id})")
            except Exception as e:
                logger.warning(f"Error fetching tracked role {role_id}: {e}")
                tracked_role_list.append(f"Unknown Role ({role_id})")
        embed.add_field(name="📊 Tracked Roles", value='\n'.join(tracked_role_list), inline=False)
    else:
        embed.add_field(name="📊 Tracked Roles", value="No tracked roles (tracking everyone)", inline=False)

    await ctx.send(embed=embed)

if __name__ == "__main__":
    # Load authorized data
    asyncio.run(load_authorized_data())

    # Load bot configuration
    load_config()

    # Get bot token from .env file
    TOKEN = os.getenv('DISCORD_TOKEN')
    if not TOKEN:
        logger.critical("DISCORD_TOKEN not found in .env file!")
        logger.critical("Please create .env file and add DISCORD_TOKEN=your_token_here.")
        exit(1)

    bot.run(TOKEN)
