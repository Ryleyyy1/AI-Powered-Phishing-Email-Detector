"""
Script untuk download dan prepare dataset phishing email
"""

import pandas as pd
import os
from pathlib import Path

# Path setup
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / 'data'
DATA_DIR.mkdir(exist_ok=True)


def create_larger_sample_dataset(output_path='data/phishing_emails_large.csv', num_samples=1000):
    """
    Generate larger sample dataset with variations
    
    Args:
        output_path (str): Path to save the dataset
        num_samples (int): Number of samples to generate
    """
    
    # Template phishing emails
    phishing_templates = [
        "URGENT! Your {service} account has been suspended. Verify at {url} immediately!",
        "Congratulations! You won ${amount}! Click {url} to claim your prize now!!!",
        "Security Alert: Suspicious activity detected. Update password at {url}",
        "Your {service} payment failed. Update billing info at {url} to avoid suspension.",
        "FINAL NOTICE: Your package is waiting. Pay ${amount} shipping fee at {url}",
        "You are pre-approved for ${amount} loan! Apply now at {url} before offer expires!",
        "Tax Refund Alert: Claim your ${amount} refund at {url}. Enter SSN to proceed.",
        "Your {service} subscription expired. Renew at {url} to continue service.",
        "VIRUS DETECTED! Download protection from {url} immediately or lose all data!",
        "Prize notification: Collect your free {item} at {url}. Limited time offer!"
    ]
    
    # Template legitimate emails
    legitimate_templates = [
        "Thank you for your order #{order_id}. Expected delivery in {days} business days.",
        "Meeting reminder: {event} on {date}. Please review the agenda beforehand.",
        "Your monthly statement is now available. Login to view account details.",
        "Welcome to our newsletter! This month featuring {topic} and upcoming events.",
        "Flight confirmation for {flight} on {date}. Check-in opens 24 hours before.",
        "Support ticket #{ticket_id} created. Our team will respond within 24-48 hours.",
        "Your subscription renews on {date}. Manage preferences in account settings.",
        "Project update: {milestone} completed. Moving to next phase next week.",
        "New course available: {course_name}. Early bird discount ends {date}.",
        "Appointment reminder: {appointment} on {date} at {time}. Please arrive early."
    ]
    
    services = ['PayPal', 'Amazon', 'Netflix', 'Bank', 'Apple ID', 'Microsoft', 'Google']
    urls = ['http://scam-site.com', 'http://phishing.com', 'http://fake-verify.com']
    amounts = ['1000', '5000', '999.99', '50000', '2500']
    items = ['iPhone', 'Laptop', 'Gift Card', 'Vacation Package']
    
    emails = []
    labels = []
    
    # Generate phishing emails
    for _ in range(num_samples // 2):
        template = phishing_templates[_ % len(phishing_templates)]
        email = template.format(
            service=services[_ % len(services)],
            url=urls[_ % len(urls)],
            amount=amounts[_ % len(amounts)],
            item=items[_ % len(items)]
        )
        emails.append(email)
        labels.append(1)
    
    # Generate legitimate emails
    for i in range(num_samples // 2):
        template = legitimate_templates[i % len(legitimate_templates)]
        email = template.format(
            order_id=f"{12345 + i}",
            days=f"{3 + (i % 5)}",
            event="Team Meeting",
            date="Next Week",
            topic="productivity tips",
            flight=f"AA{100 + i}",
            ticket_id=f"{456789 + i}",
            milestone="Phase 1",
            course_name="Introduction to AI",
            appointment="Doctor Visit",
            time="2:00 PM"
        )
        emails.append(email)
        labels.append(0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'email_text': emails,
        'label': labels
    })
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset created: {output_path}")
    print(f"   Total samples: {len(df)}")
    print(f"   Phishing: {sum(df['label'] == 1)}")
    print(f"   Legitimate: {sum(df['label'] == 0)}")
    
    return df


def download_kaggle_dataset():
    """
    Instructions untuk download dataset dari Kaggle
    """
    print("\n" + "="*70)
    print("DOWNLOAD REAL DATASET FROM KAGGLE")
    print("="*70)
    print("\nRecommended datasets:")
    print("\n1. Phishing Email Dataset")
    print("   URL: https://www.kaggle.com/datasets/subhajournal/phishingemails")
    print("   Command: kaggle datasets download -d subhajournal/phishingemails")
    
    print("\n2. Email Spam Classification Dataset")
    print("   URL: https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset")
    print("   Command: kaggle datasets download -d purusinghvi/email-spam-classification-dataset")
    
    print("\n3. CEAS 2008 Phishing Email Dataset")
    print("   Manual download from: https://monkey.org/~jose/phishing/")
    
    print("\n" + "="*70)
    print("SETUP KAGGLE API:")
    print("="*70)
    print("\n1. Install: pip install kaggle")
    print("2. Get API token from: https://www.kaggle.com/settings")
    print("3. Place kaggle.json in: ~/.kaggle/kaggle.json")
    print("4. Run download command above")
    print("\n" + "="*70)


def load_dataset(filepath='data/sample_emails.csv'):
    """
    Load and explore dataset
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    
    print("\n" + "="*70)
    print("DATASET INFORMATION")
    print("="*70)
    print(f"\nDataset: {filepath}")
    print(f"Total samples: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    print(f"\nPhishing emails: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    print(f"Legitimate emails: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("SAMPLE EMAILS")
    print("="*70)
    
    print("\n📧 Phishing Email Example:")
    print(df[df['label'] == 1]['email_text'].iloc[0][:200] + "...")
    
    print("\n📧 Legitimate Email Example:")
    print(df[df['label'] == 0]['email_text'].iloc[0][:200] + "...")
    
    print("\n" + "="*70)
    
    return df


if __name__ == "__main__":
    print("🎯 DATASET PREPARATION")
    print("="*70)
    
    # Option 1: Use existing sample dataset
    print("\n[1] Loading sample dataset...")
    try:
        df = load_dataset('data/sample_emails.csv')
    except FileNotFoundError:
        print("❌ Sample dataset not found!")
    
    # Option 2: Create larger sample dataset
    print("\n[2] Creating larger sample dataset...")
    df_large = create_larger_sample_dataset(
        output_path='data/phishing_emails_large.csv',
        num_samples=1000
    )
    
    # Option 3: Show Kaggle download instructions
    print("\n[3] Real dataset download instructions:")
    download_kaggle_dataset()
    
    print("\n✅ Dataset preparation complete!")
    print("\nNext step: Run 'python src/train_model.py' to train the model")
