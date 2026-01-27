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
        
        # get high confidence and uncertain data indexes
        high_conf_ids = np.where(confidence_score >= args.threshold)[0]
        uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), args.number_of_uncertain)[:args.number_of_uncertain]
        
        # configure label status (-1 = need human to label)
        label_status = np.zeros(confidence_score.shape)
        label_status[high_conf_ids] = 1
        label_status[uncertain_ids] = -1
        
        # create new training data (initial data + high confidence data)
        high_conf_embeddings, high_conf_labels = scaled_embeddings[high_conf_ids], result[high_conf_ids].squeeze()
        new_train_x, new_train_y = np.vstack((x_train, high_conf_embeddings)), np.hstack((y_train, high_conf_labels))
        new_train_x, new_train_y = oversampling(new_train_x, new_train_y)
        
        # train student model
        student.fit(new_train_x, new_train_y.reshape(-1, 1))
    
        # evaluate student model
        student.evaluate(x_test, y_test)
        
        # save student model
        student.save(input_sample=new_train_x[:1])
    
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
        
        # update last_batch column
        database.run_statement('update sms_spam_cd.metadata_result set last_batch = 0')
        
        # save result to mysql
        database.save_to_mysql(data_to_sql)
         
    except Exception as e:
        raise Exception(e)
    finally:
        # disconnect mysql
        database.close_connection() 


if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument('-u', '--mlflow_uri', type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument('-e', '--experiment', type=str, default='TRAIN_ON_THREE_DAY', help='name of the experiment in mlflow')
    p.add_argument('-c', '--target_column', type=str, default='spam_label', help='the column in database that indicate the type of sms (spam or ham)')
    p.add_argument('-s', '--skip_initialization', type=bool, default=True, help='whether to skip model initialization')
    p.add_argument('-n', '--number_of_uncertain', type=int, default=500, help='configure the number of uncertain message for human label')
    p.add_argument('-t', '--threshold', type=float, default=0.975, help='configure the confidence score threshold')
    args = p.parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)