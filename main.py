import sys
import mlflow
import argparse
import numpy as np
import pandas as pd 

from prefect import flow  
from src.utils.util import setup_core_components, get_unique_pattern_ids, preprocess_data, stratified_sampling, oversampling, load_train_data, load_test_data
  

@flow(name='Active Learning Pipeline')
def main(args):   
    try:
        # setup environment
        config, database, embedding_model, vectorstore, student = setup_core_components(args)   
        
        # load train data
        x_train, y_train = load_train_data(args, config, database, embedding_model, vectorstore)
        x_test, y_test = load_test_data(args, config, database, embedding_model)

        # initial teacher model if not skip
        student.fit(x_train, y_train)

        # evaluate and save teacher model
        student.evaluate(x_test, y_test)
        
        for i in range(5):
            # stratified sampling
            new_batch_data_id, new_batch_data_dt, new_batch_data_msg = stratified_sampling(config, database)

            # preprocess data
            scaled_embeddings = preprocess_data(embedding_model, new_batch_data_msg.copy(), target_column=config.data.target_column)
            
            # # reduce duplicate pattern
            # unique_pattern_id = get_unique_pattern_ids(scaled_embeddings)
            # scaled_embeddings = scaled_embeddings[unique_pattern_id]
            
            # get classification and confidence score
            result = student.predict(scaled_embeddings)
            confidence_score = student.predict_proba(scaled_embeddings)
            
            # get high confidence and uncertain data indexes
            high_conf_ids = np.where(confidence_score >= args.threshold)[0]
            uncertain_ids = np.argpartition(np.abs(confidence_score - 0.5), args.number_of_uncertain)[:args.number_of_uncertain]
            
            # create new training data (initial data + high confidence data)
            high_conf_embeddings = scaled_embeddings[high_conf_ids]
            high_conf_labels = result[high_conf_ids].squeeze()
            
            # label based on vectorstore
            uncertain_embeddings = scaled_embeddings[uncertain_ids]
            labels_status_by_vectordb, labels_by_vectordb = vectorstore.label_uncertains(new_batch_data_msg.iloc[uncertain_ids].iloc[:, 0].to_list())
            
            # configure label status (-1 = need human to label)
            label_status = np.zeros(confidence_score.shape)
            label_status[high_conf_ids] = 1
            label_status[uncertain_ids] = labels_status_by_vectordb
            
            # replace teacher model's uncertain result with labels made by vectorstore
            result[uncertain_ids] = labels_by_vectordb
            
            # combine data
            x_train, y_train = np.vstack((x_train, high_conf_embeddings, uncertain_embeddings)), np.hstack((y_train, high_conf_labels, labels_by_vectordb))
            
            # oversampling it 
            x_train, y_train = oversampling(x_train, y_train)
            
            # train student model
            student.fit(x_train, y_train.reshape(-1, 1))

            # evaluate and save student model
            student.evaluate(x_test, y_test)

            # save result to mysql
            database.save_to_mysql(
                pd.DataFrame({
                    'id': new_batch_data_id,
                    'datetime': new_batch_data_dt,
                    'spam_label': result,
                    'confidence_score': confidence_score,
                    'label_status': label_status,
                    'model': type(student).__name__
                }).to_dict(orient='records')
            )
            
            # update last_batch column
            database.run_statement(f'update sms_spam_cd.label_by_vectordb_2 set iter_involved = case when label_status in (-1, 1) then concat(coalesce(iter_involved, ""), "1") else concat(coalesce(iter_involved, ""), "0") end')
            
            
    except Exception as e:
        raise Exception(e)
    finally: 
        database.close_connection() 


if __name__ == '__main__': 
    p = argparse.ArgumentParser(description='SMS Spam Detection')
    p.add_argument('-u', '--mlflow_uri', type=str, default='http://10.168.49.12:5000', help='override mlflow tracking uri, else uses ./mlruns')
    p.add_argument('-e', '--experiment', type=str, default='TRAIN_ON_UNIQUE_PATTERN', help='name of the experiment in mlflow')
    p.add_argument('-c', '--target_column', type=str, default='spam_label', help='the column in database that indicate the type of sms (spam or ham)')
    p.add_argument('-s', '--skip_initialization', type=bool, default=True, help='whether to skip model initialization')
    p.add_argument('-n', '--number_of_uncertain', type=int, default=500, help='configure the number of uncertain message for human label')
    p.add_argument('-t', '--threshold', type=float, default=0.98, help='configure the confidence score threshold')
    args = p.parse_args()
    
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    main(args)