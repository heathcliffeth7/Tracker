# Discord Message Tracker Bot

This Discord bot tracks user message activities on your server, monitors role changes, and automatically collects Twitter/X links. It offers advanced filtering features and detailed analysis capabilities with Excel reports.

## Features

### Message Tracking
- **Role-Based Tracking**: Only tracks users with specified roles
- **Real-Time Monitoring**: Messages are recorded instantly
- **Channel-Based Filtering**: Can analyze activities in specific channels
- **Time Period Analysis**: 24 hours, 7 days, 30 days, 90 days, 180 days, 360 days and total statistics

### Content Tracking
- **Twitter/X Link Collection**: Automatically collects Twitter/X links from specified channels
- **Multi-Channel Support**: Can track content across multiple channels
- **Link Normalization**: Converts twitter.com and x.com links to standard format
- **Duplicate Prevention**: Prevents the same link from being recorded multiple times

### Role Management
- **Automatic Role Updates**: Automatically updates when user roles change
- **Audit Log Integration**: Logs role changes and responsible person
- **Advanced Role Filtering**: Complex role queries with AND, OR, NOT logic

### Reporting
- **Excel Export**: Creates detailed Excel reports
- **Multi-Time Period**: Shows all time periods in a single report
- **Hyperlink Support**: Direct access to X links
- **Filter Summary**: Detailed summary of applied filters

## Installation

### Requirements
```
discord.py
openpyxl
python-dotenv
jsonschema
```

### Installation Steps

1. **Install required packages:**
```bash
pip install discord.py openpyxl python-dotenv jsonschema
```

2. **Create `.env` file:**
```env
DISCORD_TOKEN=your_discord_bot_token_here
BOT_OWNER_ID=your_user_id_here
AUTHORIZED_USERS=user_id1,user_id2,user_id3
TRACKED_ROLE_IDS=role_id1,role_id2,role_id3
```

3. **Run the bot:**
```bash
python tracker.py
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DISCORD_TOKEN` | Discord bot token | Yes |
| `BOT_OWNER_ID` | Bot owner's user ID | Yes |
| `AUTHORIZED_USERS` | Authorized user IDs (comma-separated) | No |
| `TRACKED_ROLE_IDS` | Role IDs to track (comma-separated) | No |

### File Structure
```
├── tracker.py              # Main bot file
├── .env                    # Environment variables
├── message_tracking.json   # Message data
├── content_tracking.json   # Content data
├── bot_config.json        # Bot configuration
├── authorized_config.json  # Authorization configuration
└── bot_security.log       # Security logs
```

## Permission System

### Bot Owner vs Authorized Users

**Bot Owner:**
- Has complete control over the bot
- Can manage authorized users and roles
- Can add/remove tracked roles
- Can manage content channels
- Set via `BOT_OWNER_ID` environment variable
- Only ONE person can be the bot owner

**Authorized Users:**
- Can use bot commands (filter, stats, etc.)
- Cannot manage permissions or bot settings
- Cannot add/remove other users
- Set via `AUTHORIZED_USERS` environment variable or by bot owner
- Multiple people can be authorized users

## Commands

### Filtering Commands

#### `!filter [filters] [role_conditions] [time]`
Filters users and creates Excel report.

**Examples:**
```
!filter message>1000 content>10 @Admin 7d
!filter message>500 @Admin and @Moderator 30d
!filter content>5 @VIP or @Premium 24h
!filter message>1000 @Admin nohave @Manager 7d
!filter message>2000 roleid 123456789012345678 90d
!filter message>1000 #general 7d top10
```

**Filter Types:**
- `message>X`: More than X messages
- `content>X`: More than X content
- `@rolename`: Filter by role name
- `roleid XXXXXXXXXX`: Filter by role ID
- `#channel`: Filter by channel name
- `topX`: Top X users

**Role Operators:**
- `and`: AND logic (all roles required)
- `or`: OR logic (at least one role required)
- `nohave`: NOT logic (must not have this role)

**Time Filters:**
- `24h`: Last 24 hours
- `7d`: Last 7 days
- `30d`: Last 30 days
- `90d`: Last 90 days
- `180d`: Last 180 days
- `360d`: Last 360 days
- `total`: All time (default)

### Statistics Commands

#### `!stats`
Shows general server statistics and creates Excel report.

#### `!trackhelp`
Shows detailed list of all commands.

### Admin Commands (Bot Owner Only)

#### User Management
```
!adduser <user_id>        # Add authorized user
!removeuser <user_id>     # Remove authorized user
```

#### Role Management
```
!addrole <role_id>        # Add authorized role
!removerole <role_id>     # Remove authorized role
!addtrackedrole <role_id> # Add tracked role
!removetrackedrole <role_id> # Remove tracked role
```

#### Content Channel Management
```
!addcontentchannel <channel_id>    # Add content channel
!removecontentchannel <channel_id> # Remove content channel
!listcontentchannels               # List content channels
```

#### Listing
```
!listauth                 # List authorized users and roles
```

## Security Features

### Authorization System
- **Multi-Level Authorization**: User ID and role ID based authorization
- **Silent Protection**: Unauthorized commands are silently deleted
- **Rate Limiting**: 5-second command cooldown
- **Input Validation**: All inputs are validated

### Data Security
- **Atomic Writing**: Secure file writing that prevents data loss
- **Backup System**: Automatic backup
- **JSON Schema Validation**: Data integrity control
- **Secure Logging**: No logging of sensitive information

### Error Management
- **Graceful Degradation**: Bot continues to work in case of errors
- **Comprehensive Logging**: Detailed error logs
- **Error Hashing**: Secure error tracking

## Data Structure

### message_tracking.json
```json
{
  "user_id": {
    "username": "username",
    "current_roles": ["role1", "role2"],
    "messages": [
      {
        "timestamp": "2024-01-01T12:00:00",
        "channel_id": "123456789",
        "channel_name": "general"
      }
    ],
    "contents": [
      {
        "timestamp": "2024-01-01T12:00:00",
        "channel_id": "123456789",
        "twitter_links": ["https://x.com/user/status/123"]
      }
    ]
  }
}
```

## Important Notes

### Tracking System
- **Only users with specified roles are tracked**
- If no tracking roles are specified, all users are tracked
- Bot does not track its own messages

### Performance
- Message content is not stored for privacy and performance reasons
- Only timestamp, channel information and Twitter/X links are stored
- No automatic data cleanup, manual management required

### Limits
- Maximum 50 authorized users
- Maximum 20 authorized roles
- Maximum 20 tracked roles
- Complies with Discord API rate limits

## Troubleshooting

### Common Issues

1. **Bot not seeing messages**
   - Make sure the bot has message reading permissions in the channel
   - Check that `message_content` intent is active

2. **Commands not working**
   - Check that the user is in the authorized list
   - Make sure you're not hitting rate limit (wait 5 seconds)

3. **Role tracking not working**
   - Check that `TRACKED_ROLE_IDS` variable is set correctly
   - Make sure the bot has permission to read role information

4. **Excel file not being created**
   - Check that `openpyxl` package is installed
   - Check file write permissions

### Log Checking
```bash
tail -f bot_security.log
```

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## Support

If you encounter any issues:
1. Report the issue on GitHub Issues
2. Check log files
3. Make sure the bot has necessary permissions

---

**Note**: This bot should be used in compliance with Discord's Terms of Service. Act in accordance with local laws regarding the collection and processing of user data.
