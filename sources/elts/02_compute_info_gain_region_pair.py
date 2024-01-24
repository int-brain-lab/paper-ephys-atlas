from ephys_atlas.workflow import info_gain

t = time.time()
pids_run = report.flow.get_pids_ready('compute_raw_features', include_errors=True)
joblib.Parallel(n_jobs=18)(joblib.delayed(workflow.compute_raw_features)(pid, data_path=ROOT_PATH, path_task=ROOT_PATH) for pid in pids)
print(time.time() - t, len(pids_run))

info_gain(df_voltage, feature, mapping)