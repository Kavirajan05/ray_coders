import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
import re
from difflib import get_close_matches
import speech_recognition as sr
import pyttsx3

class WasteAnalysisEnhancer:
    """Enhanced waste analysis with monetary impact and savings calculations"""
    
    def __init__(self, data, product_mapping):
        self.data = data
        self.product_mapping = product_mapping
        self.product_costs = {
            'MILK_1L_001': 28.0 * 0.8,      # Cost = 80% of selling price
            'BREAD_LOAF_001': 35.0 * 0.8,
            'EGGS_DOZEN_001': 72.0 * 0.8,
            'RICE_1KG_001': 65.0 * 0.8,
            'OIL_1L_001': 145.0 * 0.8
        }
    
    def calculate_waste_financial_impact(self, target_year=2024):
        """Calculate detailed financial impact of waste"""
        print(f"\n💸 WASTE FINANCIAL IMPACT ANALYSIS - {target_year}")
        print("=" * 60)
        
        yearly_data = self.data[self.data['date'].dt.year == target_year]
        
        waste_analysis = {}
        total_waste_cost = 0
        
        for product_id in yearly_data['product_id'].unique():
            product_data = yearly_data[yearly_data['product_id'] == product_id]
            product_name = self.product_mapping[product_id]['name']
            
            total_wasted = product_data['stock_wasted'].sum()
            total_purchased = product_data['stock_purchased'].sum()
            waste_percentage = (total_wasted / total_purchased * 100) if total_purchased > 0 else 0
            
            # Calculate financial impact
            cost_per_unit = self.product_costs[product_id]
            waste_cost = total_wasted * cost_per_unit
            total_waste_cost += waste_cost
            
            # Calculate potential savings with different waste reduction scenarios
            waste_reduction_10 = waste_cost * 0.10  # 10% reduction
            waste_reduction_25 = waste_cost * 0.25  # 25% reduction
            waste_reduction_50 = waste_cost * 0.50  # 50% reduction
            
            waste_analysis[product_id] = {
                'product_name': product_name,
                'total_wasted_units': total_wasted,
                'waste_percentage': waste_percentage,
                'waste_cost': waste_cost,
                'cost_per_unit': cost_per_unit,
                'potential_savings_10': waste_reduction_10,
                'potential_savings_25': waste_reduction_25,
                'potential_savings_50': waste_reduction_50,
                'monthly_waste_cost': waste_cost / 12
            }
            
            print(f"📦 {product_name}")
            print(f"   🗑️ Units Wasted: {total_wasted:,} units ({waste_percentage:.1f}%)")
            print(f"   💸 Money Lost: ₹{waste_cost:,.2f}")
            print(f"   💰 Monthly Waste Cost: ₹{waste_cost/12:,.2f}")
            print(f"   📊 Potential Annual Savings:")
            print(f"      • 10% reduction: ₹{waste_reduction_10:,.2f}")
            print(f"      • 25% reduction: ₹{waste_reduction_25:,.2f}")
            print(f"      • 50% reduction: ₹{waste_reduction_50:,.2f}")
            print()
        
        print(f"💸 TOTAL WASTE COST {target_year}: ₹{total_waste_cost:,.2f}")
        print(f"📊 POTENTIAL TOTAL SAVINGS:")
        print(f"   • With 10% waste reduction: ₹{total_waste_cost * 0.10:,.2f}/year")
        print(f"   • With 25% waste reduction: ₹{total_waste_cost * 0.25:,.2f}/year")
        print(f"   • With 50% waste reduction: ₹{total_waste_cost * 0.50:,.2f}/year")
        
        return waste_analysis, total_waste_cost
    
    def generate_waste_savings_plan(self, waste_analysis, target_month):
        """Generate specific savings plan for the target month"""
        print(f"\n💡 WASTE REDUCTION SAVINGS PLAN - {calendar.month_name[target_month].upper()} 2025")
        print("=" * 65)
        
        savings_recommendations = {}
        
        for product_id, analysis in waste_analysis.items():
            product_name = analysis['product_name']
            monthly_waste_cost = analysis['monthly_waste_cost']
            
            # Generate specific savings recommendations
            recommendations = []
            
            if analysis['waste_percentage'] > 5:  # High waste
                target_reduction = 30
                monthly_savings = monthly_waste_cost * 0.30
                recommendations.append(f"🎯 TARGET: Reduce waste by {target_reduction}% to save ₹{monthly_savings:,.2f}/month")
                recommendations.append(f"💡 ACTION: Implement daily inventory checks and FIFO rotation")
                
                if product_name in ['Milk 1L Pack', 'Bread Loaf']:
                    recommendations.append(f"❄️ PERISHABLE FOCUS: Set up 2-hour freshness alerts")
                    recommendations.append(f"🏷️ MARKDOWN STRATEGY: 20% discount when 1 day to expiry")
                    
            elif analysis['waste_percentage'] > 3:  # Medium waste
                target_reduction = 20
                monthly_savings = monthly_waste_cost * 0.20
                recommendations.append(f"🎯 TARGET: Reduce waste by {target_reduction}% to save ₹{monthly_savings:,.2f}/month")
                recommendations.append(f"💡 ACTION: Weekly inventory audits and staff training")
                
            else:  # Low waste
                target_reduction = 10
                monthly_savings = monthly_waste_cost * 0.10
                recommendations.append(f"✅ GOOD PERFORMANCE: Maintain current practices")
                recommendations.append(f"🎯 OPTIMIZATION: Fine-tune by {target_reduction}% to save ₹{monthly_savings:,.2f}/month")
            
            # Add seasonal recommendations
            if target_month in [6, 7, 8, 9]:  # Monsoon
                recommendations.append(f"🌧️ MONSOON SPECIAL: Extra care for moisture-sensitive items")
            elif target_month in [10, 11]:  # Festival
                recommendations.append(f"🎉 FESTIVAL PREP: Monitor high-turnover items closely")
            
            savings_recommendations[product_id] = {
                'product_name': product_name,
                'current_monthly_waste': monthly_waste_cost,
                'target_reduction_percent': target_reduction,
                'monthly_savings_potential': monthly_savings,
                'annual_savings_potential': monthly_savings * 12,
                'recommendations': recommendations
            }
            
            print(f"🏪 {product_name.upper()}")
            print(f"   💸 Current Monthly Waste: ₹{monthly_waste_cost:,.2f}")
            print(f"   🎯 Savings Potential: ₹{monthly_savings:,.2f}/month (₹{monthly_savings*12:,.2f}/year)")
            print(f"   📋 ACTION PLAN:")
            for i, rec in enumerate(recommendations, 1):
                print(f"      {i}. {rec}")
            print()
        
        return savings_recommendations

class NaturalLanguageInterface:
    """Conversational AI interface for business queries"""
    
    def __init__(self, data, product_mapping, analytics_system):
        self.data = data
        self.product_mapping = product_mapping
        self.analytics = analytics_system
        self.product_names = [info['name'].lower() for info in product_mapping.values()]
        self.months = [calendar.month_name[i].lower() for i in range(1, 13)]
        
        # Initialize text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        
    def parse_query(self, query):
        """Parse natural language query and extract intent and entities"""
        query_lower = query.lower()
        
        # Extract product
        product_found = None
        for product_id, info in self.product_mapping.items():
            if info['name'].lower() in query_lower or any(word in query_lower for word in info['name'].lower().split()):
                product_found = (product_id, info['name'])
                break
        
        # Extract month
        month_found = None
        for i, month in enumerate(self.months, 1):
            if month in query_lower:
                month_found = (i, month.title())
                break
        
        # Extract intent
        intent = None
        if any(word in query_lower for word in ['sales', 'sold', 'selling']):
            intent = 'sales_query'
        elif any(word in query_lower for word in ['waste', 'wasted', 'spoiled']):
            intent = 'waste_query'
        elif any(word in query_lower for word in ['predict', 'forecast', 'demand']):
            intent = 'prediction_query'
        elif any(word in query_lower for word in ['why', 'reason', 'drop', 'increase', 'decrease']):
            intent = 'analysis_query'
        elif any(word in query_lower for word in ['profit', 'revenue', 'money']):
            intent = 'financial_query'
        
        return {
            'intent': intent,
            'product': product_found,
            'month': month_found,
            'original_query': query
        }
    
    def process_sales_query(self, parsed_query):
        """Process sales-related queries"""
        product_id, product_name = parsed_query['product'] if parsed_query['product'] else (None, 'all products')
        month_num, month_name = parsed_query['month'] if parsed_query['month'] else (None, 'current period')
        
        if product_id and month_num:
            # Specific product and month
            sales_data = self.data[
                (self.data['product_id'] == product_id) & 
                (self.data['month'] == month_num)
            ]
            total_sales = sales_data['quantity_sold'].sum()
            avg_daily = sales_data['quantity_sold'].mean()
            
            response = f"📊 {product_name} sales for {month_name}:\n"
            response += f"   • Total sales: {total_sales:,} units\n"
            response += f"   • Average daily sales: {avg_daily:.0f} units\n"
            response += f"   • Total revenue: ₹{total_sales * sales_data['price'].iloc[0]:,.2f}"
            
        elif product_id:
            # Specific product, all time
            sales_data = self.data[self.data['product_id'] == product_id]
            total_sales = sales_data['quantity_sold'].sum()
            monthly_avg = sales_data.groupby('month')['quantity_sold'].sum().mean()
            
            response = f"📊 {product_name} overall performance:\n"
            response += f"   • Total sales (2024): {total_sales:,} units\n"
            response += f"   • Monthly average: {monthly_avg:.0f} units\n"
            response += f"   • Total revenue: ₹{total_sales * sales_data['price'].iloc[0]:,.2f}"
            
        else:
            response = "❓ Please specify a product name (e.g., 'milk', 'bread', 'eggs')"
        
        return response
    
    def process_waste_query(self, parsed_query):
        """Process waste-related queries"""
        product_id, product_name = parsed_query['product'] if parsed_query['product'] else (None, 'all products')
        
        if product_id:
            waste_data = self.data[self.data['product_id'] == product_id]
            total_waste = waste_data['stock_wasted'].sum()
            total_purchased = waste_data['stock_purchased'].sum()
            waste_percentage = (total_waste / total_purchased * 100) if total_purchased > 0 else 0
            
            # Calculate cost
            cost_per_unit = waste_data['price'].iloc[0] * 0.8  # 80% of selling price
            waste_cost = total_waste * cost_per_unit
            
            response = f"🗑️ {product_name} waste analysis:\n"
            response += f"   • Total waste: {total_waste:,} units ({waste_percentage:.1f}%)\n"
            response += f"   • Financial impact: ₹{waste_cost:,.2f}\n"
            response += f"   • Monthly waste cost: ₹{waste_cost/12:,.2f}"
            
        else:
            total_waste = self.data['stock_wasted'].sum()
            response = f"🗑️ Overall waste: {total_waste:,} units across all products"
        
        return response
    
    def process_prediction_query(self, parsed_query):
        """Process prediction-related queries"""
        product_id, product_name = parsed_query['product'] if parsed_query['product'] else (None, None)
        
        if 'diwali' in parsed_query['original_query'].lower():
            month_num, month_name = 11, 'November'  # Diwali month
        else:
            month_num, month_name = parsed_query['month'] if parsed_query['month'] else (None, None)
        
        if product_id and month_num:
            # Use existing prediction logic
            predictions = self.analytics.predict_for_month(month_num)
            if product_id in predictions:
                pred_data = predictions[product_id]
                response = f"🔮 {product_name} prediction for {month_name} 2025:\n"
                response += f"   • Predicted sales: {pred_data['total_predicted']:,} units\n"
                response += f"   • Daily average: {pred_data['average_daily']:.0f} units\n"
                response += f"   • Recommended purchase: {int(pred_data['total_predicted'] * 1.15):,} units"
            else:
                response = f"❌ Unable to generate prediction for {product_name}"
        else:
            response = "❓ Please specify both product and time period (e.g., 'Predict milk demand for November')"
        
        return response
    
    def process_analysis_query(self, parsed_query):
        """Process why/analysis queries"""
        product_id, product_name = parsed_query['product'] if parsed_query['product'] else (None, None)
        month_num, month_name = parsed_query['month'] if parsed_query['month'] else (None, None)
        
        if product_id and month_num:
            # Get month data
            current_month = self.data[
                (self.data['product_id'] == product_id) & 
                (self.data['month'] == month_num)
            ]
            
            # Compare with previous month
            prev_month = month_num - 1 if month_num > 1 else 12
            prev_data = self.data[
                (self.data['product_id'] == product_id) & 
                (self.data['month'] == prev_month)
            ]
            
            current_sales = current_month['quantity_sold'].sum()
            prev_sales = prev_data['quantity_sold'].sum()
            change = ((current_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
            
            # Analyze reasons
            reasons = []
            if current_month['is_holiday'].sum() > prev_data['is_holiday'].sum():
                reasons.append("📅 More holidays in this month")
            if current_month['promotion_active'].sum() > prev_data['promotion_active'].sum():
                reasons.append("🎯 More promotional activities")
            if month_num in [6, 7, 8, 9]:
                reasons.append("🌧️ Monsoon season impact")
            elif month_num in [10, 11]:
                reasons.append("🎉 Festival season boost")
            
            response = f"📈 {product_name} analysis for {month_name}:\n"
            response += f"   • Sales change: {change:+.1f}% vs previous month\n"
            response += f"   • Current: {current_sales:,} units, Previous: {prev_sales:,} units\n"
            if reasons:
                response += f"   • Likely reasons: {', '.join(reasons)}"
            else:
                response += f"   • Normal seasonal variation"
        else:
            response = "❓ Please specify product and month for analysis"
        
        return response
    
    def speak_response(self, text):
        """Convert text to speech"""
        try:
            # Clean text for speech
            clean_text = re.sub(r'[📊🗑️🔮💸📈🎯📅🌧️🎉❓❌✅💡]', '', text)
            clean_text = re.sub(r'₹', 'rupees ', clean_text)
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
        except:
            print("🔊 Text-to-speech not available")
    
    def listen_for_voice_command(self):
        """Listen for voice commands"""
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                print("🎤 Listening for your question...")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=10)
            
            query = r.recognize_google(audio)
            print(f"🎤 You said: {query}")
            return query
        except sr.RequestError:
            return "❌ Could not request results from speech service"
        except sr.UnknownValueError:
            return "❌ Could not understand audio"
        except:
            return "❌ Voice recognition not available"
    
    def chat_interface(self):
        """Main chat interface"""
        print("\n🤖 RETAIL AI ASSISTANT ACTIVATED")
        print("=" * 50)
        print("💬 Ask me anything about your retail business!")
        print("📝 Examples:")
        print("   • 'Show me milk sales for July'")
        print("   • 'Why did bread sales drop in June?'")
        print("   • 'Predict rice demand for Diwali'")
        print("   • 'How much money did we waste on eggs?'")
        print("🎤 Type 'voice' for voice commands or 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                if input("\n🎤 Voice command? (y/n): ").lower().startswith('y'):
                    query = self.listen_for_voice_command()
                else:
                    query = input("\n💬 Your question: ")
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Thank you for using Retail AI Assistant!")
                    break
                
                # Parse and process query
                parsed = self.parse_query(query)
                
                if parsed['intent'] == 'sales_query':
                    response = self.process_sales_query(parsed)
                elif parsed['intent'] == 'waste_query':
                    response = self.process_waste_query(parsed)
                elif parsed['intent'] == 'prediction_query':
                    response = self.process_prediction_query(parsed)
                elif parsed['intent'] == 'analysis_query':
                    response = self.process_analysis_query(parsed)
                else:
                    response = "❓ I didn't understand that. Try asking about sales, waste, predictions, or analysis."
                
                print(f"\n🤖 AI Assistant: {response}")
                
                # Optional: speak the response
                speak_option = input("\n🔊 Should I read this aloud? (y/n): ")
                if speak_option.lower().startswith('y'):
                    self.speak_response(response)
                    
            except KeyboardInterrupt:
                print("\n👋 Chat session ended.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

class EnhancedRecommendationEngine:
    """Enhanced recommendations with natural language explanations"""
    
    def __init__(self, data, product_mapping):
        self.data = data
        self.product_mapping = product_mapping
        self.product_costs = {
            'MILK_1L_001': 28.0 * 0.8,
            'BREAD_LOAF_001': 35.0 * 0.8,
            'EGGS_DOZEN_001': 72.0 * 0.8,
            'RICE_1KG_001': 65.0 * 0.8,
            'OIL_1L_001': 145.0 * 0.8
        }
    
    def generate_natural_language_recommendations(self, target_month, predictions, comparison_data):
        """Generate detailed natural language recommendations with financial impact"""
        
        print(f"\n💬 INTELLIGENT BUSINESS RECOMMENDATIONS - {calendar.month_name[target_month].upper()} 2025")
        print("=" * 75)
        print("🤖 AI Analysis with Financial Impact and Smart Suggestions")
        print("-" * 75)
        
        for product_id, pred_data in predictions.items():
            product_name = pred_data['product_name']
            predicted_sales = pred_data['total_predicted']
            
            # Get historical data
            comp_data = comparison_data.get(product_id, {})
            prev_sales = comp_data.get('prev_sales', 0)
            prev_purchased = comp_data.get('prev_purchased', 0)
            prev_wasted = comp_data.get('prev_wasted', 0)
            prev_waste_rate = comp_data.get('prev_waste_rate', 0)
            
            # Calculate financial impact
            cost_per_unit = self.product_costs[product_id]
            prev_waste_cost = prev_wasted * cost_per_unit
            
            # Generate recommendations
            safety_stock = int(predicted_sales * 1.15)
            recommended_purchase = safety_stock
            
            # Optimize based on waste history
            if prev_waste_rate > 5:  # High waste
                optimized_purchase = int(predicted_sales * 1.10)  # Reduce safety stock
                waste_savings = (safety_stock - optimized_purchase) * cost_per_unit
                recommended_purchase = optimized_purchase
            elif prev_waste_rate > 3:  # Medium waste
                optimized_purchase = int(predicted_sales * 1.12)
                waste_savings = (safety_stock - optimized_purchase) * cost_per_unit
                recommended_purchase = optimized_purchase
            else:
                waste_savings = 0
            
            # Calculate financial projections
            investment_needed = recommended_purchase * cost_per_unit
            expected_revenue = predicted_sales * (cost_per_unit / 0.8)  # Selling price
            expected_profit = expected_revenue * 0.20
            
            print(f"\n🏪 {product_name.upper()} - SMART BUSINESS STRATEGY")
            print("─" * 55)
            
            # Historical analysis
            print(f"📊 LAST YEAR'S PERFORMANCE ANALYSIS:")
            print(f"   🛒 You purchased: {prev_purchased:,} units")
            print(f"   💰 You sold: {prev_sales:,} units")
            print(f"   🗑️ You wasted: {prev_wasted:,} units ({prev_waste_rate:.1f}%)")
            print(f"   💸 Money lost to waste: ₹{prev_waste_cost:,.2f}")
            
            # This year's strategy
            change_percent = ((predicted_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
            
            print(f"\n🎯 THIS YEAR'S SMART STRATEGY:")
            
            if change_percent > 15:
                print(f"   📈 GROWTH OPPORTUNITY: Demand expected to increase by {change_percent:.1f}%!")
                print(f"   💡 RECOMMENDATION: Since you sold {prev_sales:,} units last year,")
                print(f"      this year you should buy {recommended_purchase:,} units to capture the growth.")
                print(f"   💰 POTENTIAL: This could generate ₹{expected_profit:,.2f} profit!")
                
            elif change_percent < -15:
                print(f"   📉 MARKET DECLINE: Demand expected to decrease by {abs(change_percent):.1f}%")
                print(f"   ⚠️ CAUTION: Since you bought {prev_purchased:,} units last year and lost ₹{prev_waste_cost:,.2f} to waste,")
                print(f"      this year buy only {recommended_purchase:,} units to avoid excess inventory.")
                print(f"   💰 SAVINGS: This strategy will save you ₹{(prev_purchased - recommended_purchase) * cost_per_unit:,.2f}!")
                
            else:
                print(f"   📊 STABLE MARKET: Demand similar to last year ({change_percent:+.1f}%)")
                print(f"   💡 OPTIMIZATION: Since you bought {prev_purchased:,} units last year,")
                print(f"      this year buy {recommended_purchase:,} units for optimal efficiency.")
            
            # Waste optimization advice
            if prev_waste_rate > 3:
                potential_waste_reduction = prev_waste_cost * 0.3  # 30% reduction target
                print(f"\n🗑️ WASTE REDUCTION OPPORTUNITY:")
                print(f"   ⚠️ ALERT: You lost ₹{prev_waste_cost:,.2f} to waste last year!")
                print(f"   🎯 TARGET: Reduce waste by 30% to save ₹{potential_waste_reduction:,.2f}")
                print(f"   💡 HOW: Buy {recommended_purchase:,} units instead of {prev_purchased:,} units")
                if waste_savings > 0:
                    print(f"   💰 IMMEDIATE SAVINGS: ₹{waste_savings:,.2f} by optimizing purchase quantity")
            
            # Investment summary
            print(f"\n💼 INVESTMENT SUMMARY:")
            print(f"   💵 Investment needed: ₹{investment_needed:,.2f}")
            print(f"   💰 Expected revenue: ₹{expected_revenue:,.2f}")
            print(f"   📈 Expected profit: ₹{expected_profit:,.2f}")
            print(f"   📊 ROI: {(expected_profit/investment_needed)*100:.1f}%")
            
            # Seasonal advice
            if target_month in [6, 7, 8, 9]:
                print(f"\n🌧️ MONSOON SEASON ADVICE:")
                print(f"   • Stock 10% extra on weekends (better weather = more customers)")
                print(f"   • Keep inventory fresh - higher humidity increases spoilage risk")
                
            elif target_month in [10, 11]:
                print(f"\n🎉 FESTIVAL SEASON STRATEGY:")
                print(f"   • Prepare for 25-30% demand spike during festival weeks")
                print(f"   • Consider bulk discounts to move inventory faster")
                
            # Action plan
            print(f"\n✅ RECOMMENDED ACTION PLAN:")
            print(f"   1. 🛒 Purchase exactly {recommended_purchase:,} units")
            print(f"   2. 💰 Invest ₹{investment_needed:,.2f} for this product")
            print(f"   3. 🎯 Target selling {predicted_sales:,} units")
            print(f"   4. 📈 Expect ₹{expected_profit:,.2f} profit")
            if prev_waste_rate > 3:
                print(f"   5. 🗑️ Focus on waste reduction to save additional ₹{prev_waste_cost * 0.3:,.2f}")
            
            print("─" * 55)
        
        return True

# Integration function to add these features to the main system
def integrate_advanced_features(analytics_system):
    """Integrate all three advanced features into the main analytics system"""
    
    # Add the new features to the analytics system
    analytics_system.waste_analyzer = WasteAnalysisEnhancer(
        analytics_system.data, 
        analytics_system.product_mapping
    )
    
    analytics_system.nl_interface = NaturalLanguageInterface(
        analytics_system.data,
        analytics_system.product_mapping,
        analytics_system
    )
    
    analytics_system.enhanced_recommender = EnhancedRecommendationEngine(
        analytics_system.data,
        analytics_system.product_mapping
    )
    
    return analytics_system

# Modified main function with new features
def run_enhanced_analytics_with_new_features():
    """Run analytics system with all three new features"""
    from run_advanced_analytics import AdvancedRetailAnalytics
    
    print("🚀 ENHANCED RETAIL ANALYTICS WITH 3 NEW ADVANCED FEATURES")
    print("=" * 70)
    print("✨ NEW FEATURES:")
    print("   1. 💸 Advanced Waste Financial Analysis")
    print("   2. 🤖 Natural Language AI Assistant")
    print("   3. 💬 Enhanced Natural Language Recommendations")
    print("=" * 70)
    
    # Initialize system
    analytics = AdvancedRetailAnalytics()
    
    # Load and prepare data
    if analytics.load_and_prepare_data() is None:
        return
    
    # Integrate new features
    analytics = integrate_advanced_features(analytics)
    
    # Train models
    if not analytics.train_models():
        return
    
    # Display model accuracy
    analytics.display_model_accuracy()
    
    # NEW FEATURE 1: Advanced Waste Analysis
    waste_analysis, total_waste_cost = analytics.waste_analyzer.calculate_waste_financial_impact()
    
    # Get user input for month
    while True:
        try:
            print(f"\n🎯 MONTH SELECTION FOR 2025 PREDICTION")
            print("-" * 40)
            for i in range(1, 13):
                print(f"   {i:2d}. {calendar.month_name[i]}")
            
            month_choice = input("\n📅 Enter month number (1-12) for prediction: ").strip()
            month_choice = int(month_choice)
            
            if 1 <= month_choice <= 12:
                break
            else:
                print("❌ Please enter a number between 1 and 12")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n👋 System interrupted by user. Goodbye!")
            return
    
    # Make predictions
    predictions = analytics.predict_for_month(month_choice)
    comparison_data = analytics.compare_with_previous_year(month_choice, predictions)
    
    # NEW FEATURE 1: Generate waste savings plan
    savings_plan = analytics.waste_analyzer.generate_waste_savings_plan(waste_analysis, month_choice)
    
    # NEW FEATURE 3: Enhanced Natural Language Recommendations
    analytics.enhanced_recommender.generate_natural_language_recommendations(
        month_choice, predictions, comparison_data
    )
    
    # NEW FEATURE 2: Natural Language Interface
    chat_option = input(f"\n🤖 Would you like to chat with the AI Assistant? (y/n): ")
    if chat_option.lower().startswith('y'):
        analytics.nl_interface.chat_interface()
    
    # Save analysis
    analytics.save_analysis(month_choice, predictions, None)
    
    # Final summary
    total_investment = sum([pred['total_predicted'] * 1.15 * 
                           (28 if 'Milk' in pred['product_name'] else
                            35 if 'Bread' in pred['product_name'] else
                            72 if 'Eggs' in pred['product_name'] else
                            65 if 'Rice' in pred['product_name'] else 145) * 0.8
                           for pred in predictions.values()])
    
    print(f"\n📋 EXECUTIVE SUMMARY - {calendar.month_name[month_choice].upper()} 2025")
    print("=" * 60)
    print(f"💵 Total Investment Recommended: ₹{total_investment:,.0f}")
    print(f"💸 Previous Year Waste Cost: ₹{total_waste_cost:,.2f}")
    print(f"💰 Potential Waste Savings: ₹{total_waste_cost * 0.25:,.2f} (25% reduction)")
    print(f"🤖 AI Assistant: Ready for your questions anytime!")
    
    print(f"\n🎉 ENHANCED RETAIL ANALYTICS COMPLETE!")

if __name__ == "__main__":
    run_enhanced_analytics_with_new_features()