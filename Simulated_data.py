import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import os

# Setup
fake = Faker()
random.seed(42)
np.random.seed(42)
num_records = 656_219 # Keeping the original number of records
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Static options
resources = [
    ("/index.html", "Home View"),
    ("/jobs/placejob.php", "Job Placement"), # Assuming AI-Solutions also lists jobs or their software helps with this
    ("/jobs/viewjob.php", "Job View"),
    ("/scheduledemo.php", "Schedule Demo"),
    ("/prototype.php", "Prototype Request"), # Could link to AI Prototyping Toolkit or Custom Dev
    ("/virtualassistant.php", "AI Assistant Request"), # Direct request for the VA
    ("/event.php", "Event Info"), # Company events or webinars
    ("/images/logo.png", "Static Asset"),
    ("/solutions/dex.html", "DEX Platform Info"),
    ("/solutions/va.html", "Virtual Assistant Info"),
    ("/solutions/prototyping.html", "Prototyping Solutions Info"),
    ("/pricing.html", "Pricing Page View"),
    ("/contact.php", "Contact Us")

]
http_methods = ["GET", "POST"]
status_codes = [200, 200, 200, 304, 404, 500] # 200 more frequent
device_types = ["Desktop", "Mobile", "Tablet"]
user_types = ["New", "Returning"]
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (Linux; Android 11)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0)",
    "Mozilla/5.0 (iPad; CPU OS 13_6_1)"
]
browsers = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
campaign_sources_list = ["Email", "LinkedIn Ads", "Google Ads", "Organic Search", "Direct", "Referral Program", "Social Media", "Webinar"]
job_categories = ["Software Engineering", "AI Research", "Sales", "Marketing", "Product Management", "HR", "N/A"] # Relevant job roles

# MODIFIED SECTION STARTS HERE

# Products/Services for AI-Solutions
ai_solutions_products = [
    "DEX Platform Subscription",
    "AI Virtual Assistant License",
    "AI Prototyping Toolkit",
    "Custom AI Solution Development",
    "Innovation Accelerator Program",
    "DEX Analytics Suite",
    "Proactive Issue Resolution Service",
    "Enterprise Support & Training Package"
]

# Payment Frequencies relevant to software/services
payment_frequencies = ["Monthly Subscription", "Annual Subscription", "One-Time Purchase", "Quarterly Subscription"]

# MODIFIED SECTION ENDS HERE

start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 5, 10) # Assuming current date for generation up to recent past
time_difference = end_date - start_date

# Countries & cities
country_city_map = {
    "128": ("UK", ["London", "Manchester", "Birmingham", "Sunderland"]), # Added Sunderland
    "155": ("USA", ["New York", "San Francisco", "Chicago", "Austin"]),
    "157": ("Germany", ["Berlin", "Munich", "Hamburg", "Frankfurt"]),
    "158": ("India", ["Mumbai", "Delhi", "Bangalore", "Hyderabad"]),
    "159": ("Canada", ["Toronto", "Vancouver", "Montreal", "Ottawa"])
}
ip_blocks = list(country_city_map.keys())

# Team & salesperson structure
teams = {
    "North Europe": [fake.name() for _ in range(5)], # Example regional teams
    "Americas": [fake.name() for _ in range(5)],
    "DACH": [fake.name() for _ in range(5)], # Germany, Austria, Switzerland
    "APAC": [fake.name() for _ in range(5)]
}
team_names = list(teams.keys())

# Generator function
# Near the top of `generate_log_entry()`:


def generate_log_entry():
    random_days = random.randint(0, time_difference.days)
    random_datetime = start_date + timedelta(days=random_days, seconds=random.randint(0, 86399))
    time_str = random_datetime.time().strftime("%H:%M:%S")
    date_str = random_datetime.date().strftime("%Y-%m-%d")
    
    ip_block = random.choice(ip_blocks)
    ip = f"{ip_block}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"
    country, cities = country_city_map[ip_block]
    city = random.choice(cities)

    method = random.choice(http_methods)
    resource, action = random.choice(resources) # 'action' is the description of the resource
    status = random.choice(status_codes)
    response_time = abs(round(np.random.normal(loc=150, scale=50), 2)) # Ensure non-negative
    user_agent = random.choice(user_agents)
    device = random.choice(device_types)
    browser = random.choice(browsers)
    session_id = fake.uuid4()
    # Make referrer more realistic, sometimes it's internal
    possible_referrers = ["https://google.com", "https://bing.com", "https://linkedin.com", "-", "https://twitter.com", f"https://ai-solutions.com{random.choice(resources)[0]}"]
    referrer = random.choice(possible_referrers)
    user_type = random.choice(user_types)
    campaign = random.choice(campaign_sources_list) if random.random() > 0.3 else "N/A" # Not all traffic is from a campaign
    session_duration_minutes = round(random.uniform(1, 60), 2)  # Random session duration between 1 and 60 minutes
    # Product related fields
    # A "product" is assigned if the action is related to a direct product interaction, demo, or pricing
    # or if a POST request might signify a purchase or detailed inquiry.
    is_product_interaction = any(keyword in action.lower() for keyword in ["demo", "prototype", "assistant", "solutions", "pricing"]) or method == "POST"
    
    if is_product_interaction and status == 200 : # Assume successful interactions might lead to product interest
        product = random.choice(ai_solutions_products)
        # Adjust pricing based on product type (example logic)
        if "Development" in product or "Program" in product or "Enterprise" in product :
            premium_amount = round(random.uniform(1000, 15000), 2) # Higher for custom/enterprise
        elif "Toolkit" in product or "License" in product:
             premium_amount = round(random.uniform(200, 2000), 2) # Mid-range
        else: # Standard subscriptions/suites
            premium_amount = round(random.uniform(50, 1000), 2) # Lower for standard
        
        chosen_payment_frequency = random.choice(payment_frequencies)
        
        # PolicyEndDate (Subscription End Date) generation
        if "Subscription" in chosen_payment_frequency:
            if "Monthly" in chosen_payment_frequency:
                future_delta_days = random.randint(30, 365) # 1 month to 1 year
            elif "Quarterly" in chosen_payment_frequency:
                future_delta_days = random.randint(90, 730) # 3 months to 2 years
            elif "Annual" in chosen_payment_frequency:
                future_delta_days = random.randint(365, 1095) # 1 to 3 years
            policy_end_date = (random_datetime.date() + timedelta(days=future_delta_days)).strftime("%Y-%m-%d")
        else: # One-Time Purchase - could be N/A or a support end date
            policy_end_date = (random_datetime.date() + timedelta(days=random.randint(180, 1095))).strftime("%Y-%m-%d") # e.g. support for 6m-3yrs
            # Or set to N/A if preferred for one-time: policy_end_date = "N/A"

    else: # No specific product interaction or failed request
        product = "N/A"
        premium_amount = 0.0
        chosen_payment_frequency = "N/A"
        policy_end_date = "N/A"


    job_cat = random.choice(job_categories) if "Job" in action else "N/A"
    lead_source = random.choice(campaign_sources_list) # Can be same as campaign or different
    team = random.choice(team_names)
    salesperson = random.choice(teams[team])


    # IIS Log Format: time c-ip cs-method cs-uri-stem sc-status
    # Our extended format: Time, Date, IP, Country, City, Method, Resource, Action, Status, ResponseTimeMS, ...
    return [
        time_str, date_str, ip, country, city, method, resource, action, status, response_time,
        user_agent, device, browser, session_id, referrer, user_type, campaign, job_cat, session_duration_minutes, product,
        premium_amount, chosen_payment_frequency, policy_end_date, lead_source, team, salesperson
    ]

# Generate dataset
print(f"Generating {num_records} log entries...")
logs = [generate_log_entry() for _ in range(num_records)]

# Create DataFrame
columns = [
    "Time", "Date", "IP_Address", "Country", "City", "HTTP_Method", "Requested_Resource", "Action_Type", "HTTP_Status_Code",
    "ResponseTime_MS", "User_Agent", "Device_Type", "Browser", "Session_ID", "Referrer_URL",
    "User_Type", "Campaign_Source", "Job_Category_Interest","Session_Duration_Minutes", "Product_Service_Interest", "Deal_Value_USD", # Renamed ProductPricing
    "Payment_Frequency", "Subscription_End_Date", "Lead_Source", "Sales_Team", "Sales_Person"
]
df = pd.DataFrame(logs, columns=columns)

# Save raw (as CSV for Excel compatibility)
# The problem mentions "Excel file (CSV or similar format)"
raw_filepath = os.path.join(data_dir, 'ai_solutions_web_server_logs.csv')
df.to_csv(raw_filepath, index=False)
print(f"✅ Raw dataset saved as '{raw_filepath}'")

# Clean data
df['ResponseTime_MS'] = pd.to_numeric(df['ResponseTime_MS'], errors='coerce').fillna(0)
df['HTTP_Status_Code'] = df['HTTP_Status_Code'].astype(int)
df['Date'] = pd.to_datetime(df['Date'])
df['Deal_Value_USD'] = pd.to_numeric(df['Deal_Value_USD'], errors='coerce').fillna(0)

# Convert Subscription_End_Date to datetime, coercing errors for "N/A" values
df['Subscription_End_Date'] = pd.to_datetime(df['Subscription_End_Date'], errors='coerce')

df['Action_Type'] = df['Action_Type'].str.strip().str.lower()
df['Referrer_URL'] = df['Referrer_URL'].fillna('Unknown')
df['Hour'] = df['Time'].str.split(':').str[0].astype(int)

# Filter out server errors for cleaned dataset if desired, or handle them
# For this scenario, keeping all data might be useful for analyzing errors too.
# If strict cleaning of unsuccessful requests is needed:
# df_cleaned = df[df['HTTP_Status_Code'] < 400].copy()
df_cleaned = df.copy() # Keeping all records for now, as 404s etc can be insightful

# Save cleaned
cleaned_filepath = os.path.join(data_dir, 'cleaned_ai_solutions_web_server_logs.csv')
df_cleaned.to_csv(cleaned_filepath, index=False)
print(f"✅ Cleaned dataset saved as '{cleaned_filepath}'")

# Show sample
print("\nSample of the cleaned dataset:")
print(df_cleaned.head())

print(f"\nValue counts for Product_Service_Interest (Top 10):")
print(df_cleaned['Product_Service_Interest'].value_counts().nlargest(10))

print(f"\nValue counts for Payment_Frequency (Top 10):")
print(df_cleaned['Payment_Frequency'].value_counts().nlargest(10))

print(f"\nBasic stats for Deal_Value_USD for actual deals (value > 0):")
print(df_cleaned[df_cleaned['Deal_Value_USD'] > 0]['Deal_Value_USD'].describe())
