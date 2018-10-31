from flask import Flask, render_template, session, redirect, url_for, session
import numpy as np
from flask_wtf import FlaskForm
from wtforms import (StringField, RadioField, DecimalField, SubmitField, SelectField)
from wtforms.validators import DataRequired
import pickle

''''
For Prediction
'''
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib

app=Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'

class global_pred():
    pred_val_lr = 0
    pred_bool_lr = 'No'

class InfoForm(FlaskForm):
    '''
    This general class gets a lot of form about a person on the Titanic.
    Mainly a way to go through many of the WTForms Fields.
    '''

    age = SelectField('Age', choices=[
        ('60','< 21'),
        ('17','21 - 30'),
        ('18','31 - 40'),
        ('19','41 - 50'),
        ('20','51 - 60'),
        ('21','61 - 70'),
        ('22','71 - 80'),
        ('23','81 - 90'),
        ('24','> 90')])

    job = SelectField('Job', choices=[
        ('0','Housemaid'),
        ('1','Management'),
        ('2','Retired'),
        ('3','Student'),
        ('4','Unemployed'),
        ('60','Others')])

    marital = SelectField('Marital Status', choices=[('5','Married'),('6','Single'),('60','Divorced'),('60','Unknown')])

    education = SelectField('Education Level', choices=[('60','Primary'),('7','Secondary'),('8','Tertiary')])

    balance = SelectField('Average Yearly Balance', choices=[
        ('60','< -5000'),
        ('25','-5000 - 0'),
        ('26','1 - 5000'),
        ('27','5001 - 10000'),
        ('28','10001 - 15000'),
        ('29','15001 - 20000'),
        ('60','20001 - 25000'),
        ('30','25001 - 30000'),
        ('31','30001 - 35000'),
        ('32','35001 - 40000'),
        ('33','40001 - 45000'),
        ('34','45001 - 50000'),
        ('35','50001 - 55000'),
        ('36','55001 - 60000'),
        ('60','60001 - 65000'),
        ('37','65001 - 70000'),
        ('60','70001 - 75000'),
        ('60','75001 - 80000'),
        ('38','80001 - 85000'),
        ('60','85001 - 90000'),
        ('60','90001 - 95000'),
        ('60','95001 - 100000'),
        ('39','100001 - 105000'),
        ('60','> 105000')])

    credit_default = SelectField('Credit in Default', choices=[('60','No'),('9','Yes'),('60','Unknown')])

    housing = SelectField('Housing Loan', choices=[('60','No'),('10','Yes'),('60','Unknown')])

    loan = SelectField('Personal Loan', choices=[('60','No'),('11','Yes'),('60','Unknown')])

    contact = SelectField('Contact Method', choices=[('12','Telephone'),('13','Unknown')])

    poutcome = SelectField('Previous Campaign Outcome', choices=[('60','Failure'),('15','Success'),('14','Other'),('16','Unknown')])

    campaign = SelectField('Current Campaign Contacts', choices=[
        ('60','0 - 5'),  
        ('40','6 - 10'),
        ('41','11 - 15'),
        ('42','16 - 20'),
        ('43','21 - 25'),
        ('44','26 - 30'),
        ('45','31 - 35'),
        ('46','36 - 40'),
        ('47','41 - 45'),
        ('48','46 - 50'),
        ('60','51 - 55'),
        ('49','56 - 60'),
        ('60','> 60')])

    previous = SelectField('Previous Campaign Contacts', choices=[
        ('50','0 - 5'),  
        ('51','6 - 10'),
        ('52','11 - 15'),
        ('53','16 - 20'),
        ('54','21 - 25'),
        ('55','26 - 30'),
        ('56','31 - 35'),
        ('57','36 - 40'),
        ('60','41 - 45'),
        ('60','46 - 50'),
        ('58','51 - 55'),
        ('59','> 56')])

    submit = SubmitField('Submit')

@app.route('/', methods=['GET','POST'])
def index():
    form = InfoForm()

    test_row = np.zeros(60)

    print(test_row)

    lr_clf = joblib.load('lr_clf.joblib')

    print(lr_clf)

    if form.validate_on_submit():

        # Grab the data from the breed on the form.
        session['Age'] = form.age.data;
        if int(form.age.data) < 60:
            test_row[int(form.age.data)] = 1

        session['Job'] = form.job.data;
        if int(form.job.data) < 60:
            test_row[int(form.job.data)] = 1

        session['Marital'] = form.marital.data;
        if int(form.marital.data) < 60:
            test_row[int(form.marital.data)] = 1

        session['Education'] = form.education.data;
        if int(form.education.data) < 60:
            test_row[int(form.education.data)] = 1

        session['Balance'] = form.balance.data;
        if int(form.balance.data) < 60:
            test_row[int(form.balance.data)] = 1

        session['Default'] = form.credit_default.data;
        if int(form.credit_default.data) < 60:
            test_row[int(form.credit_default.data)] = 1

        session['Housing'] = form.housing.data;
        if int(form.housing.data) < 60:
            test_row[int(form.housing.data)] = 1

        session['Loan'] = form.loan.data;
        if int(form.loan.data) < 60:
            test_row[int(form.loan.data)] = 1

        session['Contact'] = form.contact.data;
        if int(form.contact.data) < 60:
            test_row[int(form.contact.data)] = 1

        session['Poutcome'] = form.poutcome.data;
        if int(form.poutcome.data) < 60:
            test_row[int(form.poutcome.data)] = 1

        session['Campaign'] = form.campaign.data;
        if int(form.campaign.data) < 60:
            test_row[int(form.campaign.data)] = 1

        session['Previous'] = form.previous.data;
        if int(form.previous.data) < 60:
            test_row[int(form.previous.data)] = 1

        print(test_row)

        global_pred.pred_val_lr = lr_clf.predict_proba(test_row.reshape(1, -1))[:,1][0]

        global_pred.pred_bool_lr = 'Yes' if global_pred.pred_val_lr > 0.47575057642387086 else 'No'

        print(global_pred.pred_val_lr, global_pred.pred_bool_lr)


        return redirect(url_for("predict"))

    return render_template('home.html', form=form)

@app.route('/predict')
def predict():
    prediction = global_pred.pred_val_lr
    prediction_bool = global_pred.pred_bool_lr
    return render_template('predict.html', prediction=round(prediction,2), prediction_bool=prediction_bool)

if __name__ == "__main__":
    app.run()
