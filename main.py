# main.py
from predictive_model import run_predictive_churn_pipeline
from retention_agent import deploy_retention_campaign

def main():
    print("=====================================================================")
    print("STARTING MULTIMODAL AI-POWERED CUSTOMER CHURN AND RETENTION EXECUTION")
    print("=====================================================================")
    
    # Step 1: Run tabular + text sentiment profiles through the EvoBoost Classifier
    high_risk_targets, shap_plots = run_predictive_churn_pipeline('customer_churn_400.csv')
    
    # Step 2: Stream identified at-risk customer payloads into the LangGraph state loop
    deploy_retention_campaign(high_risk_targets)
    
    print("\n=====================================================================")
    print("END-TO-END OPERATION CAMPAIGN CYCLES COMPLETED SUCCESSFULLY")
    print("=====================================================================")

if __name__ == "__main__":
    main()