Questa repo contiene tutto il necessario per ottimizzare una configurazione di parametri del framework EASE, sfruttando l'algoritmo genetico NSGAII. 
E' possibile, inoltre, effettuare la validazione dei risultati.

Il corretto utilizzo prevede:
- assicurarsi di avere il regressore correttamente trainato e il corrispondente file .joblib all'interno della cartella regressors.
- utilizzare il file dataset_generator per generare il dataset da analizzare e le corrispondenti configurazioni randomiche. Al momento, questo file prende il ingresso un dataset nel formato in cui è stato trainato il regressore e fa un sampling di 100 datapoint
- eseguire lo script NSGAII, specificando i valori di n_gen e pop_size da utilizzare. Questo script salva la configurazione ottimizzata.

Se si volesse procedere con la validazione dei risultati:
- eseguire lo script results_validation/plots_and_tables_NSGAII/results_validation.py; è necessario avere le 4 configurazioni ottimizzate generate con le differenti combinazioni di parametri, in modo da confrontarle con le configurazioni randomiche.
- lo script results_validation/comparison_with_MC/comparison_with_model_checker.py contiene tutte le successive validazioni. Lo script results_validation/comparison_with_uppaal/comparison_with_log.py permette di verificare quante configurazioni farebbero "in tempo" ad adattarsi.

E' inoltre presente uno script di conversione, per adattare il formato delle configurazioni generate a quello necessario per l'esecuzione del model checker.
