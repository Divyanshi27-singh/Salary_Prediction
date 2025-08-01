
get_ipython().system('pip install ibm-watsonx-ai | tail -n 1')
get_ipython().system('pip install autoai-libs~=2.0 | tail -n 1')
get_ipython().system("pip install -U 'lale~=0.8.3' | tail -n 1")
get_ipython().system('pip install scikit-learn==1.3.* | tail -n 1')
get_ipython().system('pip install xgboost==2.0.* | tail -n 1')
get_ipython().system('pip install lightgbm==4.2.* | tail -n 1')
get_ipython().system('pip install snapml==1.14.* | tail -n 1')


from ibm_watsonx_ai.helpers import DataConnection
from ibm_watsonx_ai.helpers import ContainerLocation

training_data_references = [
    DataConnection(
        data_asset_id='022c2c94-88b7-4e79-b84f-e0510d0d0173'
    ),
]
training_result_reference = DataConnection(
    location=ContainerLocation(
        path='auto_ml/b5d1cb78-a22b-493b-b54a-65b8441b10c7/wml_data/5fa02cdb-cb7f-41c3-893e-cc915224f184/data/automl',
        model_location='auto_ml/b5d1cb78-a22b-493b-b54a-65b8441b10c7/wml_data/5fa02cdb-cb7f-41c3-893e-cc915224f184/data/automl/model.zip',
        training_status='auto_ml/b5d1cb78-a22b-493b-b54a-65b8441b10c7/wml_data/5fa02cdb-cb7f-41c3-893e-cc915224f184/training-status.json'
    )
)


experiment_metadata = dict(
    prediction_type='regression',
    prediction_column='salary_in_usd',
    holdout_size=0.1,
    scoring='neg_root_mean_squared_error',
    csv_separator=',',
    random_state=33,
    max_number_of_estimators=2,
    training_data_references=training_data_references,
    training_result_reference=training_result_reference,
    deployment_url='https://us-south.ml.cloud.ibm.com',
    project_id='a2d55f86-d17d-4804-8486-643c0613d188',
    drop_duplicates=True,
    include_batched_ensemble_estimators=[],
    feature_selector_mode='auto'
)
import getpass

api_key = getpass.getpass("Please enter your api key (press enter): ")

from ibm_watsonx_ai import Credentials

credentials = Credentials(
    api_key=api_key,
    url=experiment_metadata['deployment_url']
)


from ibm_watsonx_ai.experiment import AutoAI

pipeline_optimizer = AutoAI(credentials, project_id=experiment_metadata['project_id']).runs.get_optimizer(metadata=experiment_metadata)

pipeline_optimizer.get_params()


summary = pipeline_optimizer.summary()
best_pipeline_name = list(summary.index)[0]
summary

pipeline_model = pipeline_optimizer.get_pipeline()

pipeline_optimizer.get_pipeline_details()['features_importance']

sklearn_pipeline_model = pipeline_optimizer.get_pipeline(astype=AutoAI.PipelineTypes.SKLEARN)

from ibm_watsonx_ai import APIClient

client = APIClient(credentials=credentials)

if 'space_id' in experiment_metadata:
    client.set.default_space(experiment_metadata['space_id'])
else:
    client.set.default_project(experiment_metadata['project_id'])

training_data_references[0].set_client(client)

from sklearn.metrics import make_scorer

from autoai_libs.scorers.scorers import neg_root_mean_squared_error

scorer = make_scorer(neg_root_mean_squared_error)

score = scorer(sklearn_pipeline_model, X_test.values, y_test.values)
print(score)

pipeline_model.visualize()

pipeline_model.pretty_print(combinators=False, ipython_display=True)


# ### Calling the `predict` method
# If you want to get a prediction by using the pipeline model object, call `pipeline_model.predict()`.

