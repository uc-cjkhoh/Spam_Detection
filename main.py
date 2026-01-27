import mlflow
import argparse
import numpy as np
import pandas as pd 

from prefect import flow 
 
from src.utils.util import setup_core_components, preprocess_data, stratified_sampling, oversampling, load_train_data, load_test_data
  

@flow(name='Active Learning Pipeline')
def main(args):   
    try:
        # setup environment
        config, database, embedding_model, vectorstore, teacher, student = setup_core_components(args)   
         
        # load train data
        x_train, y_train = load_train_data(args, config, database, embedding_model, vectorstore)
        x_test, y_test = load_test_data(args, config, database, embedding_model)
    
        # initial teacher model if not skip 
        teacher.fit(x_train, y_train) 
        
        # evaluate teacher model
        teacher.evaluate(x_test, y_test)
        
        # save teacher model
        teacher.save(input_sample=x_train[:1])
         
        # stratified sampling
        data_id, data_dt, data_msg = stratified_sampling(config, database)

        # preprocess data
        scaled_embeddings = preprocess_data(embedding_model, data_msg, target_column=config.data.target_column)
        
        # classification
        result, confidence_score = teacher.predict(scaled_embeddings), teacher.predict_proba(scaled_embeddings)
        
        # prepare data to be store in mysql
        high_conf_ids = np.where(confidence_score >= args.threshold)[0]
        uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), args.number_of_uncertain)[:args.number_of_uncertain]
        label_status = np.zeros(confidence_score.shape)
        label_status[high_conf_ids] = 1
        label_status[uncertain_ids] = -1 
        
        high_conf_embeddings, high_conf_labels = scaled_embeddings[high_conf_ids], result[high_conf_ids].squeeze()
        pseud_x, pseud_y = np.vstack((x_train, high_conf_embeddings)), np.hstack((y_train, high_conf_labels))
        pseud_x, pseud_y = oversampling(pseud_x, pseud_y)
        
        # train student model 
        student.fit(pseud_x, pseud_y.reshape(-1, 1))
    
        # evaluate student model
        student.evaluate(x_test, y_test)
        
        # save student model
        student.save(input_sample=pseud_x[:1])
    
        # save uncertainty from teacher model
        data_to_sql = pd.DataFrame({
            'id': data_id,
            'datetime': data_dt,
            'spam_label': result,
            'confidence_score': confidence_score,
            'label_status': label_status,
            'model': type(teacher).__name__,
            'last_batch': [1] * len(data_id)
        }).to_dict(orient='records')
        
        # update mysql
        database.update_db(data_to_sql)
         
    except Exception as e:
        raise Exception(e)
    finally:
        database.close_connection() 


if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument('-u', '--mlflow_uri', type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument('-e', '--experiment', type=str, default='SMS_SPAM_DETECTION_V3', help='name of the experiment in mlflow')
    p.add_argument('-c', '--target_column', type=str, default='spam_label', help='the column in database that indicate the type of sms (spam or ham)')
    p.add_argument('-s', '--skip_initialization', type=bool, default=True, help='whether to skip model initialization')
    p.add_argument('-n', '--number_of_uncertain', type=int, default=1000, help='configure the number of uncertain message for human label')
    p.add_argument('-t', '--threshold', type=float, default=0.975, help='configure the confidence score threshold')
    args = p.parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)