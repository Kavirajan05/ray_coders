"""
DATABASE INITIALIZATION SCRIPT - FIXED VERSION
===============================================
Initialize database with sample data for retail analytics
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_fixed import app, db, User, ProductData, PredictionLog, ChatHistory, AuditLog

def init_database():
    """Initialize database with tables and sample data"""
    
    with app.app_context():
        print("🔄 Initializing database...")
        
        # Drop all tables and recreate (for development)
        db.drop_all()
        db.create_all()
        
        # Create users
        create_sample_users()
        
        # Create sample product data
        create_sample_product_data()
        
        # Create sample audit logs
        create_sample_audit_logs()
        
        # Create sample chat history (moved inside app context)
        create_sample_chat_history()
        
        print("✅ Database initialized successfully!")

def create_sample_users():
    """Create sample users for testing"""
    
    print("👥 Creating sample users...")
    
    users_data = [
        {
            'username': 'admin',
            'email': 'admin@retailsight.com',
            'password': 'Admin123!',
            'role': 'admin'
        },
        {
            'username': 'retailer1',
            'email': 'retailer1@store.com',
            'password': 'Retailer123!',
            'role': 'retailer'
        },
        {
            'username': 'storemanager',
            'email': 'manager@store.com',
            'password': 'Manager123!',
            'role': 'retailer'
        },
        {
            'username': 'demo',
            'email': 'demo@retailsight.com',
            'password': 'Demo123!',
            'role': 'retailer'
        }
    ]
    
    for user_data in users_data:
        user = User(
            username=user_data['username'],
            email=user_data['email'],
            role=user_data['role']
        )
        user.set_password(user_data['password'])
        
        # Set some users as having logged in before
        if user_data['username'] != 'demo':
            user.last_login = datetime.utcnow() - timedelta(days=random.randint(1, 30))
        
        db.session.add(user)
    
    db.session.commit()
    print(f"   ✅ Created {len(users_data)} users")

def create_sample_product_data():
    """Create sample product data for the last 12 months"""
    
    print("📦 Creating sample product data...")
    
    products = {
        'MILK_1L_001': {'name': 'Milk 1L Pack', 'category': 'Dairy', 'price': 28.0},
        'BREAD_LOAF_001': {'name': 'Bread Loaf', 'category': 'Bakery', 'price': 35.0},
        'EGGS_DOZEN_001': {'name': 'Eggs (12 pieces)', 'category': 'Dairy', 'price': 72.0},
        'RICE_1KG_001': {'name': 'Rice 1KG Pack', 'category': 'Grocery', 'price': 65.0},
        'OIL_1L_001': {'name': 'Cooking Oil 1L', 'category': 'Grocery', 'price': 145.0}
    }
    
    # Base demand patterns
    base_demand = {
        'MILK_1L_001': 120,
        'BREAD_LOAF_001': 85,
        'EGGS_DOZEN_001': 70,
        'RICE_1KG_001': 50,
        'OIL_1L_001': 35
    }
    
    # Seasonal factors by month
    seasonal_factors = {
        1: 0.85, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 0.95,
        7: 0.90, 8: 0.95, 9: 1.10, 10: 1.20, 11: 1.15, 12: 1.10
    }
    
    # Generate data for last 12 months
    start_date = datetime.now().date() - timedelta(days=365)
    end_date = datetime.now().date() - timedelta(days=1)
    
    current_date = start_date
    records_created = 0
    
    while current_date <= end_date:
        month = current_date.month
        day_of_week = current_date.weekday()
        is_weekend = day_of_week >= 5
        
        for product_id, product_info in products.items():
            # Calculate demand with variations
            base = base_demand[product_id]
            seasonal = seasonal_factors[month]
            weekend_boost = 1.3 if is_weekend else 1.0
            
            # Add random variation
            demand = base * seasonal * weekend_boost * random.uniform(0.8, 1.2)
            demand = int(max(1, demand))
            
            # Calculate stock levels
            stock_purchased = int(demand * random.uniform(1.1, 1.4))
            quantity_sold = min(stock_purchased, demand)
            
            # Calculate waste (2-8% of purchased)
            waste_rate = random.uniform(0.02, 0.08)
            if product_id in ['MILK_1L_001', 'BREAD_LOAF_001']:  # Perishables
                waste_rate = random.uniform(0.05, 0.12)
            
            stock_wasted = int((stock_purchased - quantity_sold) * waste_rate)
            stock_remaining = max(0, stock_purchased - quantity_sold - stock_wasted)
            
            # Calculate financials
            revenue = quantity_sold * product_info['price']
            profit = revenue * 0.20  # 20% profit margin
            
            # Create database record
            product_data = ProductData(
                product_id=product_id,
                product_name=product_info['name'],
                category=product_info['category'],
                price=product_info['price'],
                date=current_date,
                stock_purchased=stock_purchased,
                quantity_sold=quantity_sold,
                stock_wasted=stock_wasted,
                stock_remaining=stock_remaining,
                revenue=revenue,
                profit=profit
            )
            
            db.session.add(product_data)
            records_created += 1
        
        current_date += timedelta(days=1)
    
    db.session.commit()
    print(f"   ✅ Created {records_created} product data records")

def create_sample_audit_logs():
    """Create sample audit logs"""
    
    print("📋 Creating sample audit logs...")
    
    actions = [
        'LOGIN_SUCCESS',
        'LOGIN_FAILED',
        'LOGOUT',
        'DASHBOARD_VIEW',
        'ANALYTICS_VIEW',
        'ML_PREDICTION',
        'CHAT_QUERY',
        'INVENTORY_VIEW',
        'ADMIN_VIEW'
    ]
    
    ip_addresses = [
        '192.168.1.100',
        '192.168.1.101',
        '10.0.0.50',
        '172.16.0.100'
    ]
    
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    ]
    
    # Create logs for the last 30 days
    for i in range(200):  # 200 sample log entries
        log_date = datetime.utcnow() - timedelta(days=random.randint(0, 30))
        
        audit_log = AuditLog(
            user_id=random.randint(1, 4),  # Random user ID
            action=random.choice(actions),
            details=f"Sample audit log entry {i+1}",
            ip_address=random.choice(ip_addresses),
            user_agent=random.choice(user_agents),
            created_at=log_date
        )
        
        db.session.add(audit_log)
    
    db.session.commit()
    print("   ✅ Created sample audit logs")

def create_sample_chat_history():
    """Create sample chat history"""
    
    print("💬 Creating sample chat history...")
    
    sample_chats = [
        {
            'message': 'Show me milk sales for last month',
            'response': 'Milk sales for November: 3,156 units sold, ₹88,368 revenue. Daily average: 105 units.'
        },
        {
            'message': 'Why did bread sales drop in July?',
            'response': 'Bread sales dropped in July due to monsoon season impact, reducing customer footfall by 15%.'
        },
        {
            'message': 'Predict demand for December',
            'response': 'December predictions: Milk 2,890 units, Bread 2,145 units. Festival season boost expected.'
        },
        {
            'message': 'How much money are we losing to waste?',
            'response': 'Total waste cost: ₹23,100. Highest waste: Eggs (₹7,890). Recommend FIFO rotation.'
        },
        {
            'message': 'What is our profit margin on rice?',
            'response': 'Rice 1KG Pack profit margin: 20.3%. Total profit this year: ₹39,000. Above average performance.'
        }
    ]
    
    for i, chat in enumerate(sample_chats):
        chat_entry = ChatHistory(
            user_id=random.randint(2, 4),  # Retailer users only
            message=chat['message'],
            response=chat['response'],
            created_at=datetime.utcnow() - timedelta(days=random.randint(0, 15))
        )
        
        db.session.add(chat_entry)
    
    db.session.commit()
    print(f"   ✅ Created {len(sample_chats)} chat history entries")

def verify_database():
    """Verify database initialization"""
    
    print("\n🔍 Verifying database initialization...")
    
    with app.app_context():
        user_count = User.query.count()
        product_count = ProductData.query.count()
        audit_count = AuditLog.query.count()
        chat_count = ChatHistory.query.count()
        
        print(f"   👥 Users: {user_count}")
        print(f"   📦 Product records: {product_count}")
        print(f"   📋 Audit logs: {audit_count}")
        print(f"   💬 Chat history: {chat_count}")
        
        # Test user authentication
        test_user = User.query.filter_by(username='retailer1').first()
        if test_user and test_user.check_password('Retailer123!'):
            print("   ✅ User authentication working")
        else:
            print("   ❌ User authentication failed")
        
        print("   ✅ Database verification complete")

if __name__ == '__main__':
    print("🚀 RETAIL ANALYTICS DATABASE INITIALIZATION")
    print("=" * 50)
    
    try:
        init_database()
        verify_database()
        
        print("\n🎉 Database initialization completed successfully!")
        print("\n📋 Default Login Credentials:")
        print("   Admin: admin / Admin123!")
        print("   Retailer: retailer1 / Retailer123!")
        print("   Demo: demo / Demo123!")
        print("\n🚀 Next Steps:")
        print("   1. Run: python app_fixed.py")
        print("   2. Open frontend application in browser")
        print("   3. Login with credentials above")
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)