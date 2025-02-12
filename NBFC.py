import numpy as np
import pandas as pd
from scipy.stats import norm

def calculate_default_probability(credit_score):
    """Estimate default probability based on credit score using logistic function."""
    return 1 / (1 + np.exp((credit_score - 650) / 50))  # Example function

def calculate_tranche_price(tranche_amount, tranche_yield, discount_rate, tenure):
    """
    Calculate how much a tranche can be sold for using correct present value discounting.
    """
    tenure = int(tenure)
    discount_rate = max(discount_rate, 1e-6)  # Ensure non-zero discount rate

    # Annual cash flow (interest)
    interest_payment = tranche_yield * tranche_amount

    # Discounted cash flows
    cash_flows = [interest_payment / ((1 + discount_rate) ** t) for t in range(1, tenure)]
    
    # Discounted final principal repayment
    final_payment = tranche_amount / ((1 + discount_rate) ** tenure)
    
    tranche_price = sum(cash_flows) + final_payment
    return tranche_price




def calculate_tranche_pricing(loans):
    """Pool loans into tranches based on credit score ranking and estimate pricing."""
    
    total_loan_value = loans['loan_amount'].sum()
    
    # Sort loans by credit score in descending order (Higher credit scores first)
    loans = loans.sort_values(by='credit_score', ascending=False).reset_index(drop=True)
    
    
    # Initialize tranche lists
    senior_tranche, mezzanine_tranche, equity_tranche = [], [], []
    cumulative_sum = 0
    
    # Loop through sorted loans and allocate to tranches
    for _, row in loans.iterrows():
        cumulative_sum += row['loan_amount']
        if cumulative_sum <= 0.7 * total_loan_value:
            senior_tranche.append(row)
        elif cumulative_sum <= 0.9 * total_loan_value:
            mezzanine_tranche.append(row)
        else:
            equity_tranche.append(row)
    
    # Convert lists to DataFrames
    senior_tranche = pd.DataFrame(senior_tranche)
    mezzanine_tranche = pd.DataFrame(mezzanine_tranche)
    equity_tranche = pd.DataFrame(equity_tranche)
    
    # Compute loss expected per tranche using weighted default probabilities
    senior_loss = np.dot(senior_tranche['loan_amount'], senior_tranche['default_prob']) * 0.6  # LGD = 60%
    mezzanine_loss = np.dot(mezzanine_tranche['loan_amount'], mezzanine_tranche['default_prob']) * 0.6
    equity_loss = np.dot(equity_tranche['loan_amount'], equity_tranche['default_prob']) * 0.6
    
    # Compute return expectations per tranche with a uniform base rate

    senior_yield = np.dot(senior_tranche['loan_amount'], senior_tranche['interest_rate'])/senior_tranche['loan_amount'].sum()
    mezzanine_yield = np.dot(mezzanine_tranche['loan_amount'], mezzanine_tranche['interest_rate'])/mezzanine_tranche['loan_amount'].sum()
    equity_yield = np.dot(equity_tranche['loan_amount'], equity_tranche['interest_rate'])/equity_tranche['loan_amount'].sum()
    
    # nbfc commision = 0.02% in yield
    commission = 0.02
    senior_price =    calculate_tranche_price(int(senior_tranche['loan_amount'].sum()), float(senior_yield), float(senior_yield)-commission, int(senior_tranche['tenure'].mean()))
    mezzanine_price = calculate_tranche_price(int(mezzanine_tranche['loan_amount'].sum()), float(mezzanine_yield), float(mezzanine_yield)-commission, int(mezzanine_tranche['tenure'].mean()))
    equity_price =    calculate_tranche_price(int(equity_tranche['loan_amount'].sum()), float(equity_yield), float(equity_yield)-commission, int(equity_tranche['tenure'].mean()))
    
    return {
        "Senior Tranche": {"Yield": senior_yield, "Price":senior_price, "Loss": senior_loss},
        "Mezzanine Tranche": {"Yield": mezzanine_yield, "Price":mezzanine_price, "Loss": mezzanine_loss},
        "Equity Tranche": {"Yield": equity_yield, "Price":equity_price, "Loss": equity_loss}
    }

def price_cds(loans):
    """Estimate CDS premium based on portfolio risk using weighted default probability."""
    total_loan_value = loans['loan_amount'].sum()
    weighted_default_prob = np.dot(loans['loan_amount'], loans['default_prob']) / total_loan_value
    loss_given_default = 0.6  # Assume 60% LGD
    
    cds_spread = (weighted_default_prob * loss_given_default) * 10000  # Basis points
    return cds_spread  # Annual cost per 100 notional

# Sample loan portfolio
data = {
    "credit_score": [700, 650, 600, 750, 680, 620],
    "loan_amount": [100000, 150000, 120000, 200000, 180000, 130000],
    "interest_rate": [0.07, 0.08, 0.09, 0.065, 0.075, 0.085],
    "tenure": [15, 20, 10, 30, 25, 12],
    "annual_income": [500000, 450000, 400000, 600000, 550000, 420000],
    "property_value": [1200000, 1000000, 900000, 1500000, 1300000, 950000]
}

loans = pd.DataFrame(data)

# Assign default probabilities
loans['default_prob'] = loans['credit_score'].apply(calculate_default_probability)

# Compute securitization and CDS pricing
tranche_pricing = calculate_tranche_pricing(loans)
cds_price = price_cds(loans)

# Print results
print("Total_loan", loans['loan_amount'].sum())
print("Tranche Pricing:", tranche_pricing)
print("CDS Spread (bps):", cds_price)

