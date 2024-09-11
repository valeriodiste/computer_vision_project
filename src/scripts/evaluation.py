# Import PyTorch and its modules
import torch
from torch.nn import functional
# Import other modules
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# Import custom modules
try:
	from src.scripts import datasets, models	 # type: ignore
	from src.scripts.utils import RANDOM_SEED	 # type: ignore
	from tqdm import tqdm
except ModuleNotFoundError:
	from computer_vision_project.src.scripts import datasets, models	 # type: ignore
	from computer_vision_project.src.scripts.utils import RANDOM_SEED	 # type: ignore
	from tqdm.notebook import tqdm

# Seed random number generators for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Function to compute the indexing accuracy for the given model
def compute_indexing_accuracy(indexing_dataset, retrieval_dataset, model, k_results=1, print_debug=False, **kwargs):
	
	'''
	Computes the indexing accuracy for the given model.

	Args:
	- indexing_dataset (datasets.TransformerIndexingDataset): The indexing dataset.
	- retrieval_dataset (datasets.TransformerImageRetrievalDataset): The retrieval dataset.
	- model (BoVW | DSI_VisionTransformer): The model to evaluate (either a Bag of Visual Words model or a DSI Vision Transformer model).
	- k_results (int): The number of image IDs to retrieve.
	- print_debug (bool): Whether to print debug information.

	Returns:
	- dict: The evaluation results, with structure:
		{
			"model_type": str,
			"k_results": int,
			"accuracy": float,
			"predictions": list,
			"infos": dict (optional)
		}
	'''

	# Set the model to evaluation mode
	model.eval()

	# Results dictionary
	results = {
		"model_type": model.__class__.__name__,
		"k_results": k_results,
		"accuracy": 0,
		"ids": [],
		"predictions": [],
		"infos": {}
	}
	total_correct_predictions = 0

	if print_debug:
		print("Evaluating the model to compute the indexing accuracy...")

	if isinstance(model, models.BoVW):
		print("WARNING: The BoVW model does not support indexing accuracy evaluation. Please provide a DSI Vision Transformer model instead.")
		return None

	# Compute the predictions for the indexing dataset
	for i in range(len(indexing_dataset)):

		# Get the image to test (as a tensor of the encoded image)
		test_image = torch.tensor(indexing_dataset[i][0], dtype=torch.float32).to(model.device)

		# Get the correct index for the image
		correct_indexing_id = retrieval_dataset.decode_image_id(indexing_dataset[i][1])
		correct_index = -1
		if int(correct_indexing_id) in retrieval_dataset.indexing_db_to_images_db.keys():
			correct_index = retrieval_dataset.indexing_db_to_images_db[int(correct_indexing_id)]
		elif str(correct_indexing_id) in retrieval_dataset.indexing_db_to_images_db.keys():
			correct_index = retrieval_dataset.indexing_db_to_images_db[str(correct_indexing_id)]
		correct_index = int(correct_index)

		# Get the model's prediction for the image (the encoded image ID)
		predicted_indexing_ids = model.generate_top_k_image_ids(test_image, k_results, retrieval_dataset)

		# Get the predicted index for the image	
		predicted_indexes = []
		for predicted_indexing_id in predicted_indexing_ids:
			predicted_index = -1
			if int(predicted_indexing_id) in retrieval_dataset.indexing_db_to_images_db.keys():
				predicted_index = int(retrieval_dataset.indexing_db_to_images_db[int(predicted_indexing_id)])
			elif str(predicted_indexing_id) in retrieval_dataset.indexing_db_to_images_db.keys():
				predicted_index = int(retrieval_dataset.indexing_db_to_images_db[str(predicted_indexing_id)])
			else:
				predicted_index = "ERROR: " + str(predicted_indexing_ids)
			predicted_indexes.append(predicted_index)

		# Compute the accuracy for the image
		accuracy = 1 if correct_index in predicted_indexes else 0

		# Update the total number of correct predictions
		total_correct_predictions += accuracy

		# Append the image ID, the correct index, the predicted index, and the accuracy to the results
		results["ids"].append(correct_index)
		results["predictions"].append(predicted_indexes)

		if print_debug:
			print(f"  Image {i+1}/{len(indexing_dataset)}: Correct index: {correct_index} | Predicted indexes: {predicted_indexes}")

	# Compute the accuracy
	results["accuracy"] = total_correct_predictions / len(indexing_dataset)

	return results

# Function to compute the mean average precision at k for the given model
def compute_mean_average_precision_at_k(images_db, classes, k_results=10, n_images=10, print_debug=False, model=None, retrieval_dataset=None, retrieval_test_set=None, **kwargs):
	
	'''
	Computes the precision at k for the given model.

	Args:
	- images_db (list): The images database.
	- classes (dict): The classes dictionary.
	- k_results (int): The number of relevant image IDs to retrieve.
	- n_images (int): The number of images for which to retrieve the K relevant image IDs.
	- print_debug (bool): Whether to print debug information.
	- model (BoVW | DSI_VisionTransformer): The model to evaluate (either a Bag of Visual Words model or a DSI Vision Transformer model).
	- retrieval_dataset (datasets.TransformerImageRetrievalDataset): The retrieval dataset.
	- retrieval_test_set (dict): The test set for the retrieval dataset.

	Returns:
	- dict: The evaluation results, with structure:
		{
			"model_type": str,
			"mean_average_precision": float,
			"evaluated_images": {
				image_id: precision_at_k,
				...
			},
			"k_results": int,
			"n_images": int,
			"infos": dict (optional)
		}
	'''

	# Results dictionary
	results = {
		"model_type": model.__class__.__name__,
		"mean_average_precision": 0,
		"evaluated_images": {},
		"k_results": k_results,
		"n_images": n_images,
		"infos": {}
	}

	# Re-initialize random seed
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	random.seed(RANDOM_SEED)

	# Set the model to evaluation mode
	if model is not None and not isinstance(model, models.BoVW):
		model.eval()

	# Get the test set for the retrieval dataset
	test_images = retrieval_test_set["images"]
	test_relevant_ids = retrieval_test_set["relevant_ids"]

	# Get N random image IDs from the encoded test images
	random_image_ids = random.sample(range(len(test_images)), n_images)

	# Test the trained model by generating the top k image IDs for an image in the retrieval dataset
	for i in range(n_images):

		# Get the image index of the image to test
		test_img_retrieval_db_index = random_image_ids[i]

		# Get the encoded image to test
		test_image = torch.tensor(test_images[test_img_retrieval_db_index], dtype=torch.float32)

		# Find the image ID in the retrieval dataset
		test_image_index_in_retrieval_dataset = -1
		test_image_index_in_images_db = -1
		for j in range(len(retrieval_dataset)):
			if test_image.device != retrieval_dataset[j][0].device:
				test_image = test_image.to(retrieval_dataset[j][0].device)
			if torch.all(torch.eq(retrieval_dataset[j][0], test_image)):
				test_image_index_in_retrieval_dataset = j
				test_image_index_in_images_db = retrieval_dataset.original_ids[j]
				break

		# Get the actual relevant image IDs for the image
		relevant_image_db_ids = test_relevant_ids[test_img_retrieval_db_index]
		relevant_image_ids_as_int = [int(i) if isinstance(i, str) and i.isdigit() else i for i in relevant_image_db_ids]

		# Get the top k image IDs for the first image in the retrieval dataset
		predicted_image_ids_as_int = []
		if isinstance(model, models.BoVW):
			test_image_obj = images_db[test_image_index_in_images_db]
			top_k_image_objects = model.get_similar_images(test_image_obj, k_results)
			predicted_image_ids = [img_obj["image_id"] for img_obj in top_k_image_objects]
			predicted_image_ids_as_int = [int(i) if isinstance(i, str) and i.isdigit() else i for i in predicted_image_ids]
		elif isinstance(model, models.DSI_VisionTransformer):
			test_image = test_image.to(model.device)
			top_k_image_ids = model.generate_top_k_image_ids(test_image, k_results, retrieval_dataset)
			predicted_remapped_image_ids = [retrieval_dataset.indexing_db_to_images_db[int(i)] if int(i) in retrieval_dataset.indexing_db_to_images_db.keys() else str(i) for i in top_k_image_ids]
			predicted_image_ids_as_int = [int(i) if isinstance(i, str) and i.isdigit() else i for i in predicted_remapped_image_ids]
		else:
			raise ValueError("Invalid model: " + str(model.__class__.__name__))

		# Count how many of the top K retrieved images are also in the relevant images
		num_of_correct_predictions = len(set(predicted_image_ids_as_int).intersection(relevant_image_ids_as_int))

		# Compute the precision at k for the image query
		precision = num_of_correct_predictions / min(k_results, len(predicted_image_ids_as_int))
		results["evaluated_images"][test_image_index_in_retrieval_dataset] = precision

		if print_debug:
			print(f"  Precision at K={k_results} for image {i+1}/{n_images}: {precision}")

	# Compute the mean average precision
	if results["evaluated_images"] != {}:
		results["mean_average_precision"] = np.mean(list(results["evaluated_images"].values()))

	return results

# Function to compute the recall at k for the given model
def compute_recall_at_k(images_db, classes, k_results=10, n_images=3, print_debug=False, model=None, retrieval_dataset=None, retrieval_test_set=None, **kwargs):
	'''
	Computes the recall at k for the given model (for the given number of random image queries)

	Args:
	- images_db (list): The images database.
	- classes (dict): The classes dictionary.
	- k_results (int): The number of relevant image IDs to retrieve.
	- n_images (int): The number of images for which to retrieve the K relevant image IDs.
	- print_debug (bool): Whether to print debug information.
	- model (BoVW | DSI_VisionTransformer): The model to evaluate (either a Bag of Visual Words model or a DSI Vision Transformer model).
	- retrieval_dataset (datasets.TransformerImageRetrievalDataset): The retrieval dataset.
	- retrieval_test_set (dict): The test set for the retrieval dataset.

	Returns:
	- dict: The evaluation results, with structure:
		{
			"model_type": str,
			"recall_at_k_results": list,
			"k_results": int,
			"image_ids": list,
			"infos": dict (optional)
		}
	'''

	# Re-initialize random seed
	np.random.seed(RANDOM_SEED)
	torch.manual_seed(RANDOM_SEED)
	random.seed(RANDOM_SEED)

	# Results dictionary
	results = {
		"model_type": model.__class__.__name__,
		"recall_at_k_results": [],
		"k_results": k_results,
		"image_ids": [],
		"infos": {}
	}

	# Set the model to evaluation mode
	if model is not None and not isinstance(model, models.BoVW):
		model.eval()

	# Get the test set for the retrieval dataset
	test_images = retrieval_test_set["images"]
	test_relevant_ids = retrieval_test_set["relevant_ids"]

	# Get N random image IDs from the encoded test images
	random_image_ids = random.sample(range(len(test_images)), n_images)

	# Test the trained model by generating the top k image IDs for an image in the retrieval dataset
	for i in range(n_images):

		# Get the image index of the image to test
		test_img_retrieval_db_index = random_image_ids[i]

		# Get the encoded image to test
		test_image = torch.tensor(test_images[test_img_retrieval_db_index], dtype=torch.float32)
	
		# Find the image ID in the retrieval dataset
		test_image_index_in_retrieval_dataset = -1
		test_image_index_in_images_db = -1
		for j in range(len(retrieval_dataset)):
			if test_image.device != retrieval_dataset[j][0].device:
				test_image = test_image.to(retrieval_dataset[j][0].device)
			if torch.all(torch.eq(retrieval_dataset[j][0], test_image)):
				test_image_index_in_retrieval_dataset = j
				test_image_index_in_images_db = retrieval_dataset.original_ids[j]
				break

		# Get the actual relevant image IDs for the image
		relevant_image_db_ids = test_relevant_ids[test_img_retrieval_db_index]
		relevant_image_ids_as_int = [int(i) if isinstance(i, str) and i.isdigit() else i for i in relevant_image_db_ids]

		# Get the top k image IDs for the first image in the retrieval dataset
		predicted_image_ids_as_int = []
		if isinstance(model, models.BoVW):
			test_image_obj = images_db[test_image_index_in_images_db]
			top_k_image_objects = model.get_similar_images(test_image_obj, k_results)
			predicted_image_ids = [img_obj["image_id"] for img_obj in top_k_image_objects]
			predicted_image_ids_as_int = [int(i) if isinstance(i, str) and i.isdigit() else i for i in predicted_image_ids]
		elif isinstance(model, models.DSI_VisionTransformer):
			test_image = test_image.to(model.device)
			top_k_image_ids = model.generate_top_k_image_ids(test_image, k_results, retrieval_dataset)
			predicted_remapped_image_ids = [retrieval_dataset.indexing_db_to_images_db[int(i)] if int(i) in retrieval_dataset.indexing_db_to_images_db.keys() else str(i) for i in top_k_image_ids]
			predicted_image_ids_as_int = [int(i) if isinstance(i, str) and i.isdigit() else i for i in predicted_remapped_image_ids]
		else:
			raise ValueError("Invalid model: " + str(model.__class__.__name__))

		# Count how many of the top K retrieved images are also in the relevant images
		num_of_correct_predictions = len(set(predicted_image_ids_as_int).intersection(relevant_image_ids_as_int))

		results["image_ids"].append(test_image_index_in_retrieval_dataset)
		recall = min(num_of_correct_predictions / len(relevant_image_ids_as_int), 1.0)
		results["recall_at_k_results"].append(recall)

		if print_debug:
			print(f"  Recall at K={k_results} for image {i+1}/{n_images}: {recall}")

	# Compute the mean average precision
	if results["recall_at_k_results"] != {}:
		results["mean_average_precision"] = np.mean(list(results["recall_at_k_results"]))

	return results
