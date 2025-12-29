import csv
import sys
from pathlib import Path
 
# Add the project root to the path (parent of data/ folder)
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from app import app
from auth.models import db, User
 
 
def seed_users_from_csv(csv_file_path: str = "docs/users_seed.csv", skip_existing: bool = True):
    """
    Seed users from CSV file into the database.
    Args:
        csv_file_path: Path to the CSV file containing user data
        skip_existing: If True, skip users that already exist in the database
    Returns:
        Tuple of (success_count, skip_count, error_count)
    """
    csv_path = Path(csv_file_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        return 0, 0, 0
    success_count = 0
    skip_count = 0
    error_count = 0
    with app.app_context():
        print(f"📁 Reading users from {csv_path}")
        try:
            with open(csv_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        email = row.get('email', '').strip().lower()
                        password = row.get('password_hash', '').strip()  # In CSV it's plain text
                        role = row.get('role', 'user').strip()
                        if not email or not password:
                            print("⚠️  Skipping row with missing email or password")
                            error_count += 1
                            continue
                        # Check if user already exists
                        existing_user = User.query.filter_by(email=email).first()
                        if existing_user:
                            if skip_existing:
                                print(f"⏭️  Skipping existing user: {email}")
                                skip_count += 1
                                continue
                            else:
                                print(f"🔄 Updating existing user: {email}")
                                existing_user.set_password(password)
                                existing_user.role = role
                                success_count += 1
                        else:
                            # Create new user
                            user = User(email=email, role=role)
                            user.set_password(password)
                            db.session.add(user)
                            print(f"✅ Added user: {email} (role: {role})")
                            success_count += 1
                        # Commit every 50 users to avoid memory issues
                        if (success_count + skip_count) % 50 == 0:
                            db.session.commit()
                    except Exception as e:
                        print(f"❌ Error processing row {row}: {e}")
                        error_count += 1
                        db.session.rollback()
                        continue
                # Final commit
                db.session.commit()
                print("\n" + "="*60)
                print(f"✅ Successfully added/updated: {success_count} users")
                print(f"⏭️  Skipped (already exist): {skip_count} users")
                print(f"❌ Errors: {error_count} rows")
                print("="*60)
        except Exception as e:
            print(f"❌ Error reading CSV file: {e}")
            db.session.rollback()
            return 0, 0, 1
    return success_count, skip_count, error_count
 
 
def clear_all_users():
    """Clear all users from the database (use with caution!)"""
    with app.app_context():
        try:
            count = User.query.count()
            response = input(f"⚠️  This will delete {count} users from the database. Are you sure? (yes/no): ")
            if response.lower() == 'yes':
                User.query.delete()
                db.session.commit()
                print(f"✅ Deleted {count} users from the database")
                return True
            else:
                print("❌ Operation cancelled")
                return False
        except Exception as e:
            print(f"❌ Error clearing users: {e}")
            db.session.rollback()
            return False
 
 
def list_users(limit: int = 10):
    """List users from the database"""
    with app.app_context():
        try:
            total_count = User.query.count()
            users = User.query.limit(limit).all()
            print(f"\n📊 Total users in database: {total_count}")
            print(f"Showing first {min(limit, total_count)} users:\n")
            print(f"{'ID':<5} {'Email':<40} {'Role':<10} {'Created At'}")
            print("-" * 80)
            for user in users:
                created_at = user.created_at.strftime('%Y-%m-%d %H:%M:%S') if user.created_at else 'N/A'
                print(f"{user.id:<5} {user.email:<40} {user.role:<10} {created_at}")
            if total_count > limit:
                print(f"\n... and {total_count - limit} more users")
        except Exception as e:
            print(f"❌ Error listing users: {e}")
 
 
def main():
    """Main function to run the seeding script"""
    import argparse
    parser = argparse.ArgumentParser(description='Seed users from CSV into the database')
    parser.add_argument('--csv', default='docs/users_seed.csv', help='Path to CSV file')
    parser.add_argument('--update-existing', action='store_true',
                        help='Update existing users instead of skipping them')
    parser.add_argument('--clear', action='store_true',
                        help='Clear all users before seeding')
    parser.add_argument('--list', action='store_true',
                        help='List existing users in the database')
    parser.add_argument('--list-all', action='store_true',
                        help='List all users in the database')
    args = parser.parse_args()
    # Ensure database tables exist
    with app.app_context():
        db.create_all()
        print("✅ Database tables created/verified\n")
    if args.list or args.list_all:
        limit = 1000 if args.list_all else 10
        list_users(limit)
        return
    if args.clear:
        if clear_all_users():
            print()
        else:
            return
    # Seed users
    skip_existing = not args.update_existing
    success, skipped, errors = seed_users_from_csv(args.csv, skip_existing)
    if success > 0 or skipped > 0:
        print("\n📊 Listing first 10 users:")
        list_users(10)
 
 
if __name__ == '__main__':
    main()