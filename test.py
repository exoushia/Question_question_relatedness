

def run_test(path,preprocess,vocab,model,target='class',rest_col=['id','q1_Title','q1_Body','q1_AcceptedAnswerBody',
                                                        'q1_AnswersBody','q2_Title','q2_Body','q2_AcceptedAnswerBody',
                                                        'q2_AnswersBody'], 
                                                         mapping_trimsize = {'q1_Title':10,'q1_Body':60,'answer_text1':180,'q2_Title':10,'q2_Body':60,'answer_text2':180} ):
    

    print("Initial Preprocessing on Test set: \n\n")
    if preprocess :
        print("Columns in Test  ")
        print(rest_col.append(target))
        preprocess_class = Preprocessing(path,target)
        df_test, new_cols = preprocess_class.run()
    else:
        df_test = pd.read_csv(path,usecols=rest_col.append(target))
    
    batchify_obj = forming_batches(vocab,mapping_trimsize,df_test,target,vocab_new=False)
    df_test , vocab_obj = batchify_obj.run()

    print("Initial Preprocessing completed! \n")

    print("\n\n Sequence of columns in Test Set: ")
    rest_col = [col for col in list(df_test.columns) if col not in ['id']]
    print(rest_col)
    dataset_title = Bilstm_Dataset(df_test,rest_col[0:2], rest_col[-1])
    dataset_body = Bilstm_Dataset(df_test,rest_col[2:4], rest_col[-1])
    dataset_answer = Bilstm_Dataset(df_test,rest_col[4:6], rest_col[-1])

    TEST_SIZE = len(df2_test)
    #If the batch size is none, it means no need to form batches in test ie batch size = total test size
    if config.batch_size_test is None:
        config.batch_size_test = TEST_SIZE

    num_batches_test =  int((TEST_SIZE)/config.batch_size_test)

    test_idx = np.random.choice(indices, size = TEST_SIZE, replace = False)
    test_sampler = SubsetRandomSampler(test_idx)

    test_loader_title = DataLoader(dataset_title, batch_size = config.batch_size_test, sampler = test_sampler)
    test_loader_body = DataLoader(dataset_body, batch_size = config.batch_size_test, sampler = test_sampler)
    test_loader_ans = DataLoader(dataset_answer, batch_size = config.batch_size_test, sampler = test_sampler)

    test_loaders = [test_loader_title,test_loader_body,test_loader_ans]
    print("Dataloaders for test set made! \n")

    print("Starting Evaluation on Test.. \n\n")
    test_loss , test_acc , maintaining_F1 = evaluate_model(model, test_loaders, num_batches_test)
    print("DEBUG \n")
    print("The loss from test set \n")
    print(test_loss)
    print("The F1 scores from test set \n")
    print(maintaining_F1)
    
    print("Ending Evaluation on Test.. \n\n")

    return test_loss , test_acc , maintaining_F1

