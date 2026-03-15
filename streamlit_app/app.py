# streamlit_app/app.py
import streamlit as st
import pandas as pd
from utils import load_model, predict_loan_default, get_available_models, get_model_performance

def show_input_form():

    st.subheader("📋 Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ['Sex Not Available', 'Male', 'Joint', 'Female'])
        age = st.selectbox("Age", ['25-34','35-44','45-54','55-64','65-74','>74','<25','Not Provided'])
        credit_score = st.slider("Credit Score", 500, 900, 700)
        income = st.slider("Monthly Income ($)", 0, 50000, 5000)

    with col2:
        credit_type = st.selectbox("Credit Type", ['EXP','EQUI','CRIF','CIB'])
        credit_worthiness = st.selectbox("Credit Worthiness", ['l1','l2'])
        loan_amount = st.slider("Loan Amount ($)", 10000, 1000000, 150000)
        dtir1 = st.slider("Debt-To-Income Ratio", 5.0, 60.0, 36.0)

    st.subheader("🏠 Loan Information")

    col3, col4 = st.columns(2)

    with col3:
        loan_type = st.selectbox("Loan Type", ['type1','type2','type3'])
        loan_purpose = st.selectbox("Loan Purpose", ['p1','p2','p3','p4','Not Provided'])
        interest_rate = st.slider("Interest Rate", 0.0, 10.0, 4.5)
        interest_rate_spread = st.slider("Interest Rate Spread", 0.0, 10.0, 2.0)

    with col4:
        term = st.slider("Loan Term (Months)", 96, 360, 360)
        property_value = st.slider("Property Value ($)", 10000, 1000000, 250000)
        ltv = st.slider("Loan-To-Value Ratio", 10.0, 100.0, 80.0)
        upfront_charges = st.slider("Upfront Charges", 0, 10000, 1000)

    st.subheader("⚙ Loan Structure")

    col5, col6 = st.columns(2)

    with col5:
        neg_ammortization = st.selectbox("Negative Amortization", ['not_neg','neg_amm','Not Provided'])
        interest_only = st.selectbox("Interest Only", ['not_int','int_only'])
        lump_sum_payment = st.selectbox("Lump Sum Payment", ['not_lpsm','lpsm'])

    with col6:
        approv_in_adv = st.selectbox("Pre Approved", ['nopre','pre','Not Provided'])
        submission_of_application = st.selectbox("Submission Type", ['to_inst','not_inst','Not Provided'])
        occupancy_type = st.selectbox("Occupancy Type", ['pr','sr','ir'])
        secured_by = st.selectbox("Secured By", ['Home','Land'])


    input_data = {

        "Gender": gender,
        "age": age,
        "Credit_Score": credit_score,
        "income": income,
        "credit_type": credit_type,
        "Credit_Worthiness": credit_worthiness,
        "loan_amount": loan_amount,
        "dtir1": dtir1,
        "LTV": ltv,
        "loan_type": loan_type,
        "loan_purpose": loan_purpose,
        "rate_of_interest": interest_rate,
        "Interest_rate_spread": interest_rate_spread,
        "Upfront_charges": upfront_charges,
        "term": term,
        "Neg_ammortization": neg_ammortization,
        "interest_only": interest_only,
        "lump_sum_payment": lump_sum_payment,
        "property_value": property_value,
        "occupancy_type": occupancy_type,
        "Secured_by": secured_by,
        "submission_of_application": submission_of_application,
        "approv_in_adv": approv_in_adv
    }

    return input_data

def main():
    st.set_page_config(page_title="CreditPathAI", layout="wide", initial_sidebar_state="expanded")
    
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        .main-header h1 {
            margin: 0;
            font-size: 48px;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .main-header p {
            margin: 10px 0 0 0;
            font-size: 18px;
            opacity: 0.95;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 18px;
            font-weight: 600;
            padding: 15px 40px;
            border-radius: 12px;
            border: none;
            box-shadow: 0 6px 20px rgba(102,126,234,0.4);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102,126,234,0.6);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="main-header">
            <h1>🏦 CreditPathAI</h1>
            <p>Advanced Loan Default Risk Assessment Platform</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 🎯 Model Configuration")
        available_models = get_available_models()
        if not available_models:
            st.error("⚠️ No trained models found! Please run train_models.py first.")
            return
        
        selected_model = st.selectbox("Select AI Model:", available_models, index=0)
        
        performance_df = get_model_performance()
        if performance_df is not None:
            st.markdown("### 📊 Model Performance")
            model_perf = performance_df[performance_df['Model'] == selected_model]
            if not model_perf.empty:
                perf = model_perf.iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Recall", f"{perf['Recall']:.3f}")
                with col2:
                    st.metric("F1", f"{perf['F1-score']:.3f}")
                with col3:
                    st.metric("Precision", f"{perf['Precision']:.3f}")

        st.markdown("---")
        input_data = show_input_form()

    st.markdown("## 🔮 Risk Assessment Results")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🎯 Analyze Default Risk", type="primary", use_container_width=True):
            try:
                with st.spinner("🔄 Analyzing loan application..."):
                    model = load_model(selected_model)
                    result = predict_loan_default(model, input_data)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Main Prediction Card with visual gauge
                if result['prediction_label'] == 'Default':
                    prediction_color = "#dc2626"
                    prediction_icon = "⚠️"
                    prediction_text = "HIGH DEFAULT RISK DETECTED"
                    prediction_subtext = "This application shows significant risk indicators and requires careful review"
                    recommendation = "🚫 RECOMMENDATION: DENY APPLICATION"
                    rec_color = "#dc2626"
                else:
                    prediction_color = "#10b981"
                    prediction_icon = "✅"
                    prediction_text = "LOW DEFAULT RISK"
                    prediction_subtext = "This application meets standard approval criteria with acceptable risk levels"
                    recommendation = "✓ RECOMMENDATION: APPROVE APPLICATION"
                    rec_color = "#10b981"
                
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {prediction_color}15 0%, {prediction_color}30 100%); 
                                padding: 40px; border-radius: 20px; border-left: 8px solid {prediction_color};
                                box-shadow: 0 10px 40px rgba(0,0,0,0.1); margin: 20px 0;">
                        <div style="text-align: center;">
                            <div style="font-size: 72px; margin-bottom: 20px;">{prediction_icon}</div>
                            <h1 style="color: {prediction_color}; margin: 0; font-size: 36px; font-weight: 800; 
                                       letter-spacing: 2px;">{prediction_text}</h1>
                            <p style="color: #666; font-size: 18px; margin: 15px 0 30px 0;">{prediction_subtext}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Risk Gauge Visualization
                risk_percentage = result['probability_default'] * 100
                gauge_color = "#10b981" if risk_percentage < 30 else "#f59e0b" if risk_percentage < 70 else "#dc2626"
                
                st.markdown(f"""
                    <div style="background: white; padding: 30px; border-radius: 15px; 
                                box-shadow: 0 6px 20px rgba(0,0,0,0.08); margin: 25px 0;">
                        <h3 style="text-align: center; color: #1e3c72; margin-bottom: 25px; font-size: 22px;">
                            📊 Risk Assessment Meter
                        </h3>
                        <div style="position: relative; height: 40px; background: #e5e7eb; 
                                    border-radius: 25px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                            <div style="position: absolute; height: 100%; width: {risk_percentage}%; 
                                        background: linear-gradient(90deg, {gauge_color} 0%, {gauge_color}dd 100%);
                                        border-radius: 25px; transition: width 1s ease;
                                        box-shadow: 0 2px 8px {gauge_color}80;"></div>
                            <div style="position: absolute; width: 100%; text-align: center; 
                                        line-height: 40px; font-weight: 700; color: #1e3c72; font-size: 18px;">
                                {risk_percentage:.1f}% Default Risk
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 13px; color: #666;">
                            <span>🟢 Low (0-30%)</span>
                            <span>🟡 Medium (30-70%)</span>
                            <span>🔴 High (70-100%)</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability Breakdown
                st.markdown("### 📈 Probability Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%); 
                                    padding: 25px; border-radius: 15px; text-align: center; color: white;
                                    box-shadow: 0 6px 20px rgba(220,38,38,0.3); height: 160px; display: flex; 
                                    flex-direction: column; justify-content: center;">
                            <div style="font-size: 16px; opacity: 0.95; margin-bottom: 10px; 
                                        text-transform: uppercase; letter-spacing: 1px;">Default Risk</div>
                            <div style="font-size: 48px; font-weight: 800; margin: 10px 0;">
                                {result['probability_default']:.1%}
                            </div>
                            <div style="font-size: 14px; opacity: 0.9;">Likelihood of Default</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                    padding: 25px; border-radius: 15px; text-align: center; color: white;
                                    box-shadow: 0 6px 20px rgba(16,185,129,0.3); height: 160px; display: flex; 
                                    flex-direction: column; justify-content: center;">
                            <div style="font-size: 16px; opacity: 0.95; margin-bottom: 10px; 
                                        text-transform: uppercase; letter-spacing: 1px;">Approval Chance</div>
                            <div style="font-size: 48px; font-weight: 800; margin: 10px 0;">
                                {result['probability_no_default']:.1%}
                            </div>
                            <div style="font-size: 14px; opacity: 0.9;">Likelihood of Repayment</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 25px; border-radius: 15px; text-align: center; color: white;
                                    box-shadow: 0 6px 20px rgba(102,126,234,0.3); height: 160px; display: flex; 
                                    flex-direction: column; justify-content: center;">
                            <div style="font-size: 16px; opacity: 0.95; margin-bottom: 10px; 
                                        text-transform: uppercase; letter-spacing: 1px;">Confidence</div>
                            <div style="font-size: 48px; font-weight: 800; margin: 10px 0;">
                                {result['confidence']:.1%}
                            </div>
                            <div style="font-size: 14px; opacity: 0.9;">Model Certainty</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Final Recommendation Box
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, {rec_color}20 0%, {rec_color}35 100%); 
                                padding: 30px; border-radius: 15px; border: 3px solid {rec_color};
                                margin: 25px 0; text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                        <h2 style="color: {rec_color}; margin: 0; font-size: 28px; font-weight: 800; 
                                   text-transform: uppercase; letter-spacing: 2px;">{recommendation}</h2>
                        <p style="color: #666; margin: 15px 0 0 0; font-size: 16px;">
                            Based on comprehensive risk analysis using {selected_model}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #667eea15 0%, #764ba230 100%); 
                            padding: 40px; border-radius: 15px; text-align: center; margin: 30px 0;
                            border: 2px dashed #667eea;">
                    <div style="font-size: 64px; margin-bottom: 20px;">🎯</div>
                    <h3 style="color: #1e3c72; margin: 0 0 15px 0;">Ready to Analyze</h3>
                    <p style="color: #666; font-size: 16px; margin: 0;">
                        Configure applicant details in the sidebar and click<br/>
                        <strong>'Analyze Default Risk'</strong> to generate comprehensive risk assessment
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#666; font-size:14px;'>CreditPathAI © 2024 | Powered by Advanced Machine Learning</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()