import pandas as pd
from Phase2 import Two
from Phase3 import Three
from Phase1a import A
from Phase1b import B
from Phase1c import C
from Prepros import Pre

# Phase 4 (calculating the risk score)
def risk_score_calculator(df_two, df_three):
    # Merge outputs
    merged = pd.merge(df_two[['unit_number', 'failure_probability']],
                      df_three[['unit_number', 'predicted_rul']],
                      on='unit_number')

    # Calculate and normalize risk score
    merged['risk_score'] = merged['failure_probability'] / (merged['predicted_rul'] + 1e-6)
    merged['normalized_risk'] = (merged['risk_score'] - merged['risk_score'].min()) / \
                                (merged['risk_score'].max() - merged['risk_score'].min())

    # Plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.bar(merged['unit_number'], merged['normalized_risk'], color='tomato')
    plt.xlabel('Engine ID')
    plt.ylabel('Normalized Risk Score')
    plt.title('🚨 Risk Score per Engine')
    plt.axhline(0.7, color='black', linestyle='--', label='High-Risk Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

# USING THE Pre-processor class to get all the data sets arranged and with values required as per the different uses
p=Pre()
train1,test1,rul1=p.reg_a()
train2,test2,rul2=p.reg_b()
train3,test3,rul3=p.reg_c()

  # Clustering of the different datasets
# clustering of data for all the data-sets combined
a=p.pre_a_cl()
clus_a=A(a)
clus_a.cluster_a()

# clustering of data for data-sets 1 and 3 only
b1,b=p.pre_b_cl()
clus_b=B(b,b1)
clus_b.cluster_b()

# clustering of data for data-sets 2 and 4 only
c,c1=p.pre_c_cl()
clus_c=C(c,c1)
clus_c.cluster_c()


# calculating for dataset A (Phase1,Phase2,Phase3 and Phase4)
# phase 2
p_A=Two(train1.copy(),test1.copy(),rul1.copy())
A_1=p_A.model()
# phase 3
regression_A = Three(train1.copy(), test1.copy(), rul1.copy())
A_2=regression_A.next_failure_prediction()
# calculating the risk score
risk_score_calculator(A_1,A_2)


# # calculating for dataset B (Phase1 and Phase3)
# phase 2
p_B=Two(train2.copy(),test2.copy(),rul2.copy())
B_1=p_B.model()
# phase 3
regression_B = Three(train2.copy(), test2.copy(), rul2.copy())
B_2=regression_B.next_failure_prediction()
# calculating the risk score
risk_score_calculator(B_1,B_2)


# # calculating for dataset C (Phase2 and Phase4)
p_C=Two(train3.copy(),test3.copy(),rul3.copy())
C_1=p_C.model()
# phase 3
regression_C = Three(train3.copy(), test3.copy(), rul3.copy())
C_2=regression_C.next_failure_prediction()
# calculating the risk score
risk_score_calculator(C_1,C_2)
