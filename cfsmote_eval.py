import pandas as pd
import sys
import datetime
import random
from river import stream, evaluate, tree
import river.metrics
from river_fairness_metrics.metrics import Metrics as FMetrics
from river_fairness_metrics.equalized_odds import Equalized_FPR
from river_fairness_metrics.equal_opportunity import Equal_Opportunity
from river_fairness_metrics.demographic_parity import Demographic_Parity
from river_fairness_metrics.disparate_impact import Disparate_Impact
from river_fairness_metrics.fairness_unawareness import Fairness_Unawareness
from cfsmote import CFSMOTE
from pathlib import Path

def run_exp_fairness_adult(file_in, file_out, fnlwgt=False):


    converter = {'income': float}

    if fnlwgt:
        converter['fnlwgt'] = float



    category_list=['education','workclass','marital_status','occupation', 'relationship', 'race', 'native_country', 'age' 'capital-gain', 'capital-loss', 'hours-per-week']

    #WARNING! Make sure gaps for sensitive attributes work for the dataset!!!
    model = CFSMOTE(sensitive_attribute="sex", deprived_val="Female", undeprived_vals=["Male"], model_categories=category_list, num_neighbours=3, min_size_allowed=10, minority_threshold=0.245, model_name='HoeffdingAdaptiveTree')
    
    X_y = stream.iter_csv(file_in, target="income", converters=converter)

    sens_att = ("sex", "Female")

    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)

    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall()
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision()
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1()

    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))
    indv_fairness = Fairness_Unawareness(sens_att)

    results = {}

    
    for m in metrics:
        results[f"{m.__class__.__name__}"] = []
    for f in fair_metrics:
        results[f"{f.__class__.__name__}"] = []
    results['IndividualFairness'] = []
    results['time'] = []
    
    

    for x, y in X_y:
        y_pred = model.predict_one(x)

        x_switched = x.copy()
        if (x[sens_att[0]] == sens_att[1]):
            x_switched[sens_att[0]] = random.choice(model.undeprived_vals)
        else:
            x_switched[sens_att[0]] = sens_att[1]

        y_switched = model.predict_one(x_switched)

        model.learn_one(x, y)

        if y_pred is not None:
            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)

            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[f"{m.__class__.__name__}"].append(m.get())

            for f in fair_metrics:
                results[f"{f.__class__.__name__}"].append(f.get())
            
            indv_fairness.update(x=x, y_opp_pred=y_switched, y_pred=y_pred)
            results['IndividualFairness'].append(indv_fairness.get())

            results['time'].append(datetime.datetime.now())



    df = pd.DataFrame(results)
    df.to_csv(file_out)

def run_exp_fairness_student(file_in, file_out):

    converter = {'absences': float}


    category_list=['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']

    #WARNING! Make sure gaps for sensitive attributes work for the dataset!!!
    model = CFSMOTE(sensitive_attribute="sex", deprived_val="M", undeprived_vals=["F"], model_categories=category_list, num_neighbours=3, min_size_allowed=10, minority_threshold=0.245, model_name='HoeffdingAdaptiveTree')
    
    X_y = stream.iter_csv(file_in, target="income", converters=converter)

    sens_att = ("sex", "Female")

    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)

    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall()
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision()
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1()

    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))
    indv_fairness = Fairness_Unawareness(sens_att)

    results = {}

    
    for m in metrics:
        results[f"{m.__class__.__name__}"] = []
    for f in fair_metrics:
        results[f"{f.__class__.__name__}"] = []
    results['IndividualFairness'] = []
    results['time'] = []
    
    

    for x, y in X_y:
        y_pred = model.predict_one(x)

        x_switched = x.copy()
        if (x[sens_att[0]] == sens_att[1]):
            x_switched[sens_att[0]] = random.choice(model.undeprived_vals)
        else:
            x_switched[sens_att[0]] = sens_att[1]

        y_switched = model.predict_one(x_switched)

        model.learn_one(x, y)

        if y_pred is not None:
            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)

            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[f"{m.__class__.__name__}"].append(m.get())

            for f in fair_metrics:
                results[f"{f.__class__.__name__}"].append(f.get())
            
            indv_fairness.update(x=x, y_opp_pred=y_switched, y_pred=y_pred)
            results['IndividualFairness'].append(indv_fairness.get())

            results['time'].append(datetime.datetime.now())



    df = pd.DataFrame(results)
    df.to_csv(file_out)

def run_exp_fairness_kdd(file_in, file_out):

    converter = {'age_numeric': float, 'wage-per-hour': float, 'dividends-from-stocks': float, 'capital-gains': float, 'capital-losses': float, 'num-persons-worked-for-employer': float, 'weeks-worked-in-year': float, 'class': float}

    category_list = ['class-of-worker',	'detailed-industry-recode',	'detailed-occupation-recode',	'education', 'enroll-in-edu-inst-last-wk',	'marital-stat',	'major-industry-code'	,'major-occupation-code	race',	'hispanic-origin',	'sex', 	'member-of-a-labor-union' ,	'reason-for-unemployment',	'full-or-part-time-employment-stat',  'tax-filer-stat', 	'region-of-previous-residence', 	'state-of-previous-residence', 	'detailed-household-and-family-stat', 	'detailed-household-summary-in-household', 	'migration-code-change-in-msa' ,	'migration-code-change-in-reg', 'migration-code-move-within-reg',	'live-in-this-house-1-year-ago',	'migration-prev-res-in-sunbelt' , 'family-members-under-18'	, 'country-of-birth-father' ,	'country-of-birth-mother' ,	'country-of-birth-self' ,	'citizenship' ,	'own-business-or-self-employed',	'fill-inc-questionnaire-for-veterans-admin',	'veterans-benefits',	'year']


    #WARNING! Make sure gaps for sensitive attributes work for the dataset!!!
    model = CFSMOTE(sensitive_attribute="sex", deprived_val="Female", undeprived_vals=["Male"], model_categories=category_list, num_neighbours=3, min_size_allowed=10, minority_threshold=0.245, model_name='HoeffdingAdaptiveTree')
    
    X_y = stream.iter_csv(file_in, target="class", converters=converter)

    sens_att = ("sex", "Female")

    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)

    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall()
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision()
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1()

    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))
    indv_fairness = Fairness_Unawareness(sens_att)

    results = {}

    
    for m in metrics:
        results[f"{m.__class__.__name__}"] = []
    for f in fair_metrics:
        results[f"{f.__class__.__name__}"] = []
    results['IndividualFairness'] = []
    results['time'] = []
    
    

    for x, y in X_y:
        y_pred = model.predict_one(x)

        x_switched = x.copy()
        if (x[sens_att[0]] == sens_att[1]):
            x_switched[sens_att[0]] = random.choice(model.undeprived_vals)
        else:
            x_switched[sens_att[0]] = sens_att[1]

        y_switched = model.predict_one(x_switched)

        model.learn_one(x, y)

        if y_pred is not None:
            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)

            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[f"{m.__class__.__name__}"].append(m.get())

            for f in fair_metrics:
                results[f"{f.__class__.__name__}"].append(f.get())
            
            indv_fairness.update(x=x, y_opp_pred=y_switched, y_pred=y_pred)
            results['IndividualFairness'].append(indv_fairness.get())

            results['time'].append(datetime.datetime.now())



    df = pd.DataFrame(results)
    df.to_csv(file_out)



def run_exp_fairness_nypd(file_in, file_out):

    converter = {'ADDR_PCT_CD': float, 'PD_CD' : float, 'JURISDICTION_CODE': float, 'X_COORD_CD' : float, 'Y_COORD_CD': float, 'LAW_CAT_CD': float}


    category_list=['CRM_ATPT_CPTD_CD', 'BORO_NM', 'LOC_OF_OCCUR_DESC', 'PREM_TYP_DESC', 'JURIS_DESC', 'PARKS_NM',  'SUSP_AGE_GROUP', 'SUSP_RACE', 'SUSP_SEX', 'PATROL_BORO']

    #WARNING! Make sure gaps for sensitive attributes work for the dataset!!!
    model = CFSMOTE(sensitive_attribute='SUSP_SEX', deprived_val="M", undeprived_vals=["F"], model_categories=category_list, num_neighbours=3, min_size_allowed=10, minority_threshold=0.245, model_name='HoeffdingAdaptiveTree')
    
    X_y = stream.iter_csv(file_in, target="LAW_CAT_CD", converters=converter)

    sens_att = ('SUSP_SEX', "M")

    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)

    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall()
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision()
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1()

    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))
    indv_fairness = Fairness_Unawareness(sens_att)

    results = {}

    
    for m in metrics:
        results[f"{m.__class__.__name__}"] = []
    for f in fair_metrics:
        results[f"{f.__class__.__name__}"] = []
    results['IndividualFairness'] = []
    results['time'] = []
    
    

    for x, y in X_y:
        y_pred = model.predict_one(x)

        x_switched = x.copy()
        if (x[sens_att[0]] == sens_att[1]):
            x_switched[sens_att[0]] = random.choice(model.undeprived_vals)
        else:
            x_switched[sens_att[0]] = sens_att[1]

        y_switched = model.predict_one(x_switched)

        model.learn_one(x, y)

        if y_pred is not None:
            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)

            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[f"{m.__class__.__name__}"].append(m.get())

            for f in fair_metrics:
                results[f"{f.__class__.__name__}"].append(f.get())
            
            indv_fairness.update(x=x, y_opp_pred=y_switched, y_pred=y_pred)
            results['IndividualFairness'].append(indv_fairness.get())

            results['time'].append(datetime.datetime.now())



    df = pd.DataFrame(results)
    df.to_csv(file_out)


def run_exp_fairness_synth(file_in, file_out):

    converter = {'attr_1': float, 'attr_2': float, 'attr_3': float, 'attr_4': float, 'class': float}

    categorical_features = ['SA']



    #WARNING! Make sure gaps for sensitive attributes work for the dataset!!!
    model = CFSMOTE(sensitive_attribute="SA", deprived_val="Female", undeprived_vals=["Male"], model_categories=categorical_features, num_neighbours=3, min_size_allowed=10, minority_threshold=0.245, model_name='HoeffdingAdaptiveTree')
    
    X_y = stream.iter_csv(file_in, target="class", converters=converter)

    sens_att = ("SA", "Female")

    eq_fpr = Equalized_FPR(sens_att)
    eq_opp = Equal_Opportunity(sens_att)
    disp_imp = Disparate_Impact(sens_att)
    dem_parity = Demographic_Parity(sens_att)

    fair_metrics = FMetrics((dem_parity, disp_imp, eq_opp, eq_fpr))

    acc = river.metrics.Accuracy()
    bAcc = river.metrics.BalancedAccuracy()
    recall = river.metrics.Recall()
    kappa = river.metrics.CohenKappa()
    precision = river.metrics.Precision()
    gmean = river.metrics.GeometricMean()
    f1 = river.metrics.F1()

    metrics = river.metrics.base.Metrics((acc, bAcc, recall, kappa, precision, gmean, f1))
    indv_fairness = Fairness_Unawareness(sens_att)

    results = {}

    
    for m in metrics:
        results[f"{m.__class__.__name__}"] = []
    for f in fair_metrics:
        results[f"{f.__class__.__name__}"] = []
    results['IndividualFairness'] = []
    results['time'] = []
    
    

    for x, y in X_y:
        if (float(y) == -1.0): #class conversion
            y = 0

        y_pred = model.predict_one(x)

        x_switched = x.copy()
        if (x[sens_att[0]] == sens_att[1]):
            x_switched[sens_att[0]] = random.choice(model.undeprived_vals)
        else:
            x_switched[sens_att[0]] = sens_att[1]

        y_switched = model.predict_one(x_switched)

        model.learn_one(x, y)

        if y_pred is not None:
            fair_metrics.update(y_pred=y_pred, y_true=y, x=x)

            metrics.update(y_pred=y_pred, y_true=y)

            for m in metrics:
                results[f"{m.__class__.__name__}"].append(m.get())

            for f in fair_metrics:
                results[f"{f.__class__.__name__}"].append(f.get())
            
            indv_fairness.update(x=x, y_opp_pred=y_switched, y_pred=y_pred)
            results['IndividualFairness'].append(indv_fairness.get())

            results['time'].append(datetime.datetime.now())



    df = pd.DataFrame(results)
    df.to_csv(file_out)


    
def run_test_fairness(mode, repetition):


    print("Run " + str(repetition))

    print("KDD")

    in_file = f"./data/cfsmote_paper/Data_CFSMOTE/kdd/run_{repetition}.csv"
    out_file = f"./data/cfsmote_paper/results_kdd/{mode}/run_{repetition}.csv"
    run_exp_fairness_kdd(in_file, out_file)

    print("Adult")

    in_file = f"./data/cfsmote_paper/Data_CFSMOTE/adult/run_{repetition}.csv"
    out_file = f"./data/cfsmote_paper/results_adult/{mode}/run_{repetition}.csv"
    run_exp_fairness_adult(in_file, out_file)





if __name__=='__main__':
    i = int(sys.argv[1])

    
    if i == 0:
        in_file = f"./data/Data_CFSMOTE/originals/synthetic.csv"
        out_file = f"./data/Data_CFSMOTE/results_synthetic/cfsmote.csv"
        #run_exp_fairness_synth(in_file, out_file)

        in_file = f"./data/Data_CFSMOTE/originals/nypd.csv"
        out_file = f"./data/Data_CFSMOTE/results_nypd/cfsmote.csv"
        run_exp_fairness_nypd(in_file, out_file)
    

    run_test_fairness('cfsmote', i)
