# Import PyTorch and its modules
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.nn import Transformer
from torch.nn import functional
# Import PyTorch Lightning
import pytorch_lightning as pl
# Import other modules
import random
import math
import numpy as np
# Import modules for the BoVW model
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans
# Import custom modules
try:
	from src.scripts import datasets	 # type: ignore
	from src.scripts.utils import RANDOM_SEED, get_image_from_b64_string	 # type: ignore
	from tqdm import tqdm
except ModuleNotFoundError:
	from computer_vision_project.src.scripts import datasets	 # type: ignore
	from computer_vision_project.src.scripts.utils import RANDOM_SEED, get_image_from_b64_string 	# type: ignore
	from tqdm.notebook import tqdm

# Seed random number generators for reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Bag of Visual Words (BoVW) Model
class BoVW():

	def __init__(self, all_images, indexed_images, kmeans_clusters=150):
		'''
		Initializes the Bag of Visual Words (BoVW) model.

		Args:
			images_db: list of images represented as objects with various attributes.
			kmeans_clusters: Number of clusters for the k-means clustering algorithm to use for the BoVW model.
		'''
		# Initialize the images database
		self.all_images = all_images
		self.indexed_images = indexed_images
		# Initialize the number of clusters for the k-means clustering algorithm
		self.kmeans_clusters = kmeans_clusters
		# Get the Bag of Visual Words (BoVW) features of the images
		bovw_features_infos = self.get_db_bovw_features(indexed_images)
		self.bovw_features = bovw_features_infos[0]
		self.visual_words = bovw_features_infos[1]
		self.ids_mapping = bovw_features_infos[2]

	def compute_sift_features(self, images, print_debug = True):
		'''
		Extracts SIFT features from the images.
		Args:
			images: Dictionary that holds the images as <class, images> pairs
		Returns:
			Array with two elements: the first element is the list of all computed SIFT descriptors, the second element is a dictionary containing 
			<class, descriptors> pairs (thus SIFT descriptors grouped by class).
		'''
		sift_vectors = {}
		descriptors_list = []
		sift = cv2.xfeatures2d.SIFT_create()
		for key, value in tqdm(images.items(), desc="Extracting SIFT features...", disable=(len(images.items()) <= 1 or not print_debug)):
			features = []
			for img in value:
				keypoints, descriptors = sift.detectAndCompute(img,None)
				descriptors_list.extend(descriptors)
				features.append(descriptors)
			sift_vectors[key] = features
		return [descriptors_list, sift_vectors]

	def compute_bovw(self, sift_features, centers, print_debug = True):
		'''
		Creates the Bag of Visual Words (BoVW) representation for the images.
		Args:
			all_bovw: Dictionary that holds the SIFT descriptors separated by class
			centers: Array that holds the central points (visual words) of the k-means clustering
		Returns:
			Dictionary that holds the histograms for each image separated by class.
		'''
		# Auxiliary function to find the index of the closest central point to the given SIFT descriptor
		def find_index(image, center):
			'''
			Find the index of the closest central point to the given SIFT descriptor.
			Args:
				image: SIFT descriptor.
				center: Array of central points (visual words) of the k-means clustering.
			Returns:
				Index of the closest central point.
			'''
			count = 0
			ind = 0
			for i in range(len(center)):
				if(i == 0):
					count = distance.euclidean(image, center[i]) 
				else:
					dist = distance.euclidean(image, center[i]) 
					if(dist < count):
						ind = i
						count = dist
			return ind
		# Compute the Bag of Visual Words (BoVW) representation for the images
		dict_feature = {}
		for label, features in tqdm(sift_features.items(), desc="Creating BoVW representation...", disable=(len(sift_features.items()) <= 1 or not print_debug)):
			category = []
			for img in features:
				histogram = np.zeros(len(centers))
				for feature in img:
					ind = find_index(feature, centers)
					histogram[ind] += 1
				category.append(histogram)
			dict_feature[label] = category
		# Return a dictionary that holds the histograms for each image, separated by image class
		return dict_feature
	
	def get_db_bovw_features(self, images, print_debug = True):

		'''
		Extracts the Bag of Visual Words (BoVW) features from the images in the given DB.
		Args:
			images_db: List of images represented as objects with various attributes.
		Returns:
			Tuple with 3 elements: the first element is the BoVW features of the images, grouped by class, the second element is the list of visual words (centers), 
				and the third is a mapping from the BoVW features to the image IDs (as a <label, list(image ID)> dictionary).
		'''

		# Takes as input a set of images and returns the Bag of Visual Words (BoVW) representation

		# Step 0: prepare the images, to finally have a dictionary with <label, images> pairs (each image is a CV2 grayscale image)
		images_dict = {}
		ids = {}
		for i in tqdm(range(len(images)), desc="Loading BoVW images...", disable=(len(images) <= 1 or not print_debug)):
			image_obj = images[i]
			image = get_image_from_b64_string(image_obj["image_data"])
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			label = image_obj["image_label_name"]
			if label not in images_dict:
				images_dict[label] = []
			images_dict[label].append(image)
			if label not in ids:
				ids[label] = []
			ids[label].append(i)

		# Step 1: Extract SIFT features from the images
		sift_features = self.compute_sift_features(images_dict, print_debug)

		# Step 2: Perform k-means clustering to get the visual words
		if print_debug: print("Performing k-means clustering...")
		kmeans = KMeans(n_clusters = self.kmeans_clusters, n_init=10)
		kmeans.fit(sift_features[0])
		visual_words = kmeans.cluster_centers_ 

		# Step 3: Create the BoVW representation for the images
		bovw_features = self.compute_bovw(sift_features[1], visual_words, print_debug)

		if print_debug: print("BoVW features extracted.")

		# Return the BoVW features
		return bovw_features, visual_words, ids
	
	def get_bovw_features(self, image):
		'''
		Extracts the Bag of Visual Words (BoVW) features from the given image, using the words from the images database created for the BoVW model.
		Args:
			image: Image represented as an object with various attributes.
		Returns:
			BoVW features of the image.
		'''
		# Get the image as a CV2 grayscale image
		image = get_image_from_b64_string(image["image_data"])
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# Extract SIFT features from the image
		sift_features = self.compute_sift_features({"image": [image]})
		# Create the BoVW representation for the image
		bovw_features = self.compute_bovw(sift_features[1], self.visual_words)
		# Return the BoVW features
		return bovw_features["image"][0]
	
	def get_similar_images(self, image, num_images=5):
		'''
		Finds the most similar images to the given image in the images database.
		Args:
			image: Image represented as an object with various attributes.
			num_images: Number of similar images to find.
		Returns:
			List of the most similar images to the given image.
		'''
		# Get the BoVW features of the given image
		image_bovw = self.get_bovw_features(image)
		# Find the most similar images to the given image
		similar_images = []
		def cosine_similarity(a, b):
			return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
		for label, features in self.bovw_features.items():
			for i in range(len(features)):
				similarity = cosine_similarity(image_bovw, features[i])
				similar_images.append({"label": label, "similarity": similarity, "image_id": self.ids_mapping[label][i]})
		similar_images = sorted(similar_images, key=lambda x: x["similarity"], reverse=True)
		num_images = min(num_images, len(similar_images))
		return similar_images[:num_images]

# Vision Transformer Lightining Module
class DSI_VisionTransformer(pl.LightningModule):

	def __init__(self, **model_kwargs):
		super().__init__()
		self.save_hyperparameters(model_kwargs)
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# self.device = torch.device("cuda")
		self.model = DSI_ViT(**model_kwargs).to(self.device)
		# Store the outputs for training and validation steps
		self.training_losses = []
		self.validation_losses = []
		self.training_accuracies = []
		self.validation_accuracies = []

	def forward(self, imgs, ids):
		# Expects as input a tensor of shape [B, C, H, W] where:
		# - B = batch size (number of images in the batch)
		# - C = number of channels (e.g. 3 channels for RGB)
		# - H = height of the image
		# - W = width of the image
		return self.model(imgs, ids).to(self.device)

	def configure_optimizers(self):
		# Set the optimizer (use optim.Adam as the default optimizer)
		optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
		# Set the learning rate scheduler
		# lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
		lr_scheduler = None		# No learning rate scheduler used
		if lr_scheduler is not None:
			return [optimizer], [lr_scheduler]
		elif optimizer is not None:
			return optimizer
		else:
			return None

	def training_step(self, batch, batch_idx):
		# Training step for the model (compute the loss and accuracy)
		loss, accuracy = self.model.step(batch)
		# Append the loss to the training losses list (for logging)
		self.training_accuracies.append(accuracy)
		# Append the accuracy to the training accuracies list (for logging)
		self.training_losses.append(loss)
		# Return the loss
		return loss

	def validation_step(self, batch, batch_idx):
		# Validation step for the model (compute the loss and accuracy)
		loss, accuracy = self.model.step(batch, True)
		# Append the loss to the validation losses list (for logging)
		self.validation_losses.append(loss)
		# Append the accuracy to the validation accuracies list (for logging)
		self.validation_accuracies.append(accuracy)
		# Return the loss
		return loss
	
	# PyTorch Lightning function (optional) called at the very end of each training epoch
	def on_train_epoch_end(self):
		# If the validation losses list is NOT empty, return (to avoid logging the training losses twice)
		if len(self.validation_losses) > 0:
			return
		print_train_info = False
		epoch_num = self.current_epoch
		if print_train_info: print()
		# Log the average training loss for this epoch
		if not len(self.training_losses) == 0:
			avg_epoch_training_loss = torch.stack(self.training_losses).mean()
			self.log("avg_epoch_training_loss", avg_epoch_training_loss)
			if print_train_info: print(f"Average training loss for epoch {epoch_num}: ", avg_epoch_training_loss.item())
			self.training_losses.clear()
		# Log the average training accuracy for this epoch
		if not len(self.training_accuracies) == 0:
			avg_epoch_training_accuracy = torch.stack(self.training_accuracies).mean()
			self.log("avg_epoch_training_accuracy", avg_epoch_training_accuracy)
			if print_train_info: print(f"Average training accuracy for epoch {epoch_num}: ", avg_epoch_training_accuracy.item())
			self.training_accuracies.clear()

	# Pytorch lightning function (optional) called at the very end of each validation epoch
	def on_validation_epoch_end(self):
		print_val_info = False
		epoch_num = self.current_epoch
		if print_val_info: print()
		# Log the average training loss for this epoch
		if not len(self.training_losses) == 0:
			avg_epoch_training_loss = torch.stack(self.training_losses).mean()
			self.log("avg_epoch_training_loss", avg_epoch_training_loss)
			if print_val_info: print(f"Average training loss for epoch {epoch_num}: ", avg_epoch_training_loss.item())
			self.training_losses.clear()
		# Log the average validation loss for this epoch
		if not len(self.validation_losses) == 0:
			avg_epic_validation_loss = torch.stack(self.validation_losses).mean()
			self.log("avg_epoch_val_loss", avg_epic_validation_loss)
			if print_val_info: print(f"Average validation loss for epoch {epoch_num}: ", avg_epic_validation_loss.item())
			self.validation_losses.clear()
		# Log the average training accuracy for this epoch
		if not len(self.training_accuracies) == 0:
			avg_epoch_training_accuracy = torch.stack(self.training_accuracies).mean()
			self.log("avg_epoch_training_accuracy",avg_epoch_training_accuracy)
			if print_val_info: print(f"Average training accuracy for epoch {epoch_num}: ",avg_epoch_training_accuracy.item())
			self.training_accuracies.clear()
		# Log the average validation accuracy for this epoch
		if not len(self.validation_accuracies) == 0:
			avg_epoch_validation_accuracy = torch.stack(self.validation_accuracies).mean()
			self.log("avg_epoch_val_accuracy", avg_epoch_validation_accuracy)
			if print_val_info: print(f"Average validation accuracy for epoch {epoch_num}: ",avg_epoch_validation_accuracy.item())
			self.validation_accuracies.clear()

	def generate_top_k_image_ids(self, encoded_image: torch.Tensor, k: int, retrieval_dataset: datasets.TransformerRetrievalDataset, force_debug_output = False, recover_malformed_img_ids = True):
		''' 
		Generate the top K image IDs for the given image (as a tensor of shape [C, H, W])
		'''
		# Initialize random seed for reproducibility
		torch.manual_seed(RANDOM_SEED)
		# Special tokens of the image IDs encoding
		img_id_start_token = retrieval_dataset.img_id_start_token
		img_id_end_token = retrieval_dataset.img_id_end_token
		img_id_padding_token = retrieval_dataset.img_id_padding_token
		image_id_skip_token = -1	# This value is set to tokens of image ID sequences that should not be considered for the final top K results (because their predicted softmax probabilities are below a threshold)
		# Max length of the image IDs
		img_id_max_length = retrieval_dataset.img_id_max_len	# The maximum number of digits in the image ID plus 2 (for the start and end tokens)
		# Initialize the output sequence (sequence of image ID tokens, i.e. digits) as a tensor containing only the start token
		output_sequences = torch.tensor([[img_id_start_token]], dtype=torch.long, device=encoded_image.device)
		# Initialize the source sequence (image encoding) as the input image tensor
		source_sequence = encoded_image.unsqueeze(0)	# From shape [C, H, W] to shape [1, C, H, W], where 1 is considered the batch size
		# Iterate over the maximum length of the sequences (i.e. the number of tokens to generate for each image IDs)
		for i in range(img_id_max_length):
			# Repeat or reduce the source sequence to match the number of sequences in the output_sequences tensor
			model_input_source = source_sequence.repeat(output_sequences.size(0), 1, 1, 1)	# Shape: [O, C, H, W], with O being the number of sequences in the output_sequences tensor (i.e. total good image ID sequences kept till now)
			# Get the next tokens logits (no softmax used for the model's output) from the transformer model (list of N floats, with N being the number of possible target tokens, hence the 10 possible digits of image IDs)
			outputs = self(model_input_source, output_sequences)
			# Get the next token to append to each sequence (i.e. the token with the highest probability for each of the k sequences)
			sorted_logits, sorted_indices = torch.sort(outputs[-1], descending=True, dim=-1)
			# Transform the logits into probabilities using the softmax function
			probabilities = functional.softmax(sorted_logits, dim=-1)
			# Replace tokens with a probability lower than a threshold with a special token (image_id_skip_token), and keep only the top n tokens
			max_tokens_to_keep = max(0, (4 - i*2) + int(math.log10(k))) + 1	# The number of tokens to keep is a function of the iteration number and the number of top k image IDs to keep
			max_tokens_to_keep = min(max_tokens_to_keep, self.hparams.num_classes)	# The number of tokens to keep cannot be higher than the number of classes
			probability_threshold = (1.0 / self.hparams.num_classes) * 0.75	
			# Check if all the tokens have a probability lower than the threshold
			best_digits = None	# Shape: [max_tokens_to_keep]
			if torch.all(probabilities < probability_threshold):
				# If all the filtered indices are the image_id_skip_token, keep only the top n tokens
				best_digits = sorted_indices[: max_tokens_to_keep]
			else:
				# Filter out the tokens with a probability lower than the threshold and keep only the top n tokens
				best_digits = sorted_indices.masked_fill(probabilities < probability_threshold, image_id_skip_token)[: max_tokens_to_keep]
			# Remove all the best digits that are the image_id_skip_token
			best_digits = best_digits[best_digits != image_id_skip_token]
			# Repeat the target sequences to match the number of sequences in the sorted indices tensor
			output_sequences = output_sequences.repeat(best_digits.size(0), 1)
			# Reshape the best digits tensor to match the new target sequences tensor
			best_digits = best_digits.unsqueeze(1).repeat(1, output_sequences.size(0) // best_digits.size(0)).view(-1, 1)
			# Concatenate the target sequences with the sorted indices to create the new target sequences
			output_sequences = torch.cat((output_sequences, best_digits), dim=1)
		# Remove all the sequences that only contain the special tokens (i.e. the start and end tokens)
		top_image_ids_tokens = output_sequences.tolist()
		# Convert the top k sequences of image IDs' tokens to a list of k image IDs
		top_image_ids = []
		for i in range(min(k, len(top_image_ids_tokens))):
			image_id_tokens = top_image_ids_tokens[i]
			image_id = retrieval_dataset.decode_image_id(image_id_tokens, force_debug_output, recover_malformed_img_ids)
			if image_id is not None and len(image_id) > 0:
				top_image_ids.append(image_id)
		# Remove duplicate image IDs
		top_image_ids = list(set([str(idx) for idx in top_image_ids]))
		# Refill the list to have k image IDs
		image_ids_to_add = retrieval_dataset.get_similar_image_ids(k - len(top_image_ids), target_image_ids=top_image_ids)
		if force_debug_output:
			top_image_ids = top_image_ids + ["R=" + str(image_id) for image_id in image_ids_to_add]
		else:
			top_image_ids = top_image_ids + [str(image_id) for image_id in image_ids_to_add]
		# Return the top k image IDs
		return top_image_ids[0: k]


# Positional Encoding module for the Vision Transformer model (in case of Non-Learnable Positional Encodings)
class PositionalEncoding(nn.Module):

	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10_000):
		'''
		Constructor of the PositionalEncoding class (custom torch.nn.Module).

		This module implements the positional encoding module of the traditional Transformer architecture.

		For more details: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
		'''
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0)]
		x = self.dropout(x)
		return x


# DSI Vision Transformer model (Torch module, to use with the PyTorch Lightning module "DSI_VisionTransformer" above)
class DSI_ViT(nn.Module):

	def __init__(
		self,
		# Main parameters
		embed_dim,
		hidden_dim,
		num_channels,
		num_heads,
		num_layers,
		batch_size,
		num_classes,
		patch_size,
		num_patches,
		learn_positional_encodings,
		# Other parameters
		img_id_max_length,
		img_id_start_token = 10,
		img_id_end_token = 12,
		img_id_padding_token = 11,
		# Training parameters
		dropout=0.0,
		learning_rate=1e-4,
		# Other parameters (for e.g. logging purposes)
		**kwargs
	):
		"""Vision Transformer.

		Args:
			embed_dim: Dimensionality of the input feature vectors to the Transformer (i.e. the size of the embeddings)
			hidden_dim: Dimensionality of the hidden layer in the feed-forward networks within the Transformer
			num_channels: Number of channels of the input (e.g. 3 for RGB, 1 for grayscale, ecc...)
			num_heads: Number of heads to use in the Multi-Head Attention block
			num_layers: Number of layers to use in the Transformer
			batch_size: Number of samples in a batch
			num_classes: Number of classes to predict
				(in my case, since I give an image with, concatenated, the N digits of the image ID, the num_classes is the number of possible digits of the image IDs, hence 10+3, including the special tokens)
			patch_size: Number of pixels that the patches have per dimension
			num_patches: Maximum number of patches an image can have
			learn_positional_encodings: Whether to learn the positional encodings or use fixed ones
			img_id_max_length: Maximum number of digits in the image ID (including the start and end tokens)
			img_id_start_token: Token representing the start of the image ID
			img_id_end_token: Token representing the end of the image ID
			img_id_padding_token: Token representing the padding of the image ID
			dropout: Amount of dropout to apply in the feed-forward network and on the input encoding
			learning_rate: Learning rate for the optimizer
		"""
		super().__init__()

		self.patch_size = patch_size

		self.num_classes = num_classes

		self.embed_dim = embed_dim
		self.img_id_max_length = img_id_max_length
		self.img_id_start_token = img_id_start_token
		self.img_id_end_token = img_id_end_token
		self.img_id_padding_token = img_id_padding_token

		self.learn_positional_encodings = learn_positional_encodings

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Layers/Networks
		self.input_layer = nn.Linear(		# Convert the input image's patches into embeddings, i.e. vectors (one for each patch) of size "embed_dim"
			num_channels * (patch_size**2), # Input size: number of channels (3 if RGB is used) times the number of total pixels in a patch (i.e. the size of the patch)
			embed_dim,
			device=self.device
		)
		self.input_layer = self.input_layer.to(self.device)
		self.id_embedding = nn.Embedding(	# Embedding layer for the image ID digits (the 10 digits [0-9] plus the 3 special tokens, i.e. end of sequence, padding, start of sequence)
			num_classes, # The maximum number of digits in the image ID
			embed_dim,
			padding_idx=img_id_padding_token,	# The padding index is the index of the digit that represents the padding (i.e. the digit that is used to pad the image ID to the maximum length)
			device=self.device
		)
		self.id_embedding = self.id_embedding.to(self.device)
		self.transformer = nn.Sequential(
			# Add the specified number of Attention Blocks to the Transformer ("num_layers" times)
			*(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
		)
		self.transformer = self.transformer.to(self.device)
		self.mlp_head = nn.Sequential(
			nn.LayerNorm(embed_dim, device=self.device),
			nn.Linear(embed_dim, num_classes, device=self.device)
		)
		self.mlp_head = self.mlp_head.to(self.device)
		self.dropout = nn.Dropout(dropout)
		self.dropout = self.dropout.to(self.device)

		# Parameters/Embeddings
		if self.learn_positional_encodings:
			self.pos_embedding = nn.Parameter(
				torch.randn(1, num_patches+img_id_max_length+2, embed_dim, device=self.device).to(self.device)
			)	# Positional encoding for the image ID embeddings
			self.pos_embedding = self.pos_embedding.to(self.device)
		else:
			self.pos_embedding = PositionalEncoding(embed_dim, dropout=dropout)
			self.pos_embedding = self.pos_embedding.to(self.device)

	def img_to_patch(self, x : torch.Tensor, patch_size, flatten_channels=True):
		"""
		Args:
			x: Tensor representing the image of shape [B, C, H, W]
			patch_size: Number of pixels per dimension of the patches (integer)
			flatten_channels: If True, the patches will be returned in a flattened format as a feature vector instead of a image grid.

		Returns:
			x: The input image tensor reshaped into a tensor (list) of P patches, where each patch is a vector of size C*patch_size*patch_size

		"""
		B, C, H, W = x.shape	# B is the batch size (number of images in the batch), C is the number of channels (e.g. 3 channels for RGB), H is the height of the image, and W is the width of the image
		P = patch_size 			# Width and height of the patches
		H_P = H // P			# Number of patches vertically
		W_P = W // P			# Number of patches horizontally
		x = x.reshape(B, C, H_P, P, W_P, P) # [B, C, H, W] -> [B, C, H_P, P, W_P, P]	-> Reshape the image into patches
		x = x.permute(0, 2, 4, 1, 3, 5)		# [B, H_P, W_P, C, P, P]	-> Rearrange the patches so that they are in the correct order
		x = x.flatten(1, 2)  				# [B, H_P*W_P, C, P, P]		-> Flatten each patch into a vector
		if flatten_channels:
			x = x.flatten(2, 4)  			# [B, H_P*W_P, C*P*P]		-> Flatten all the patches into a single vector
		# Convert the data type to float32
		x = x.to(self.device, dtype=torch.float)
		return x

	def forward(self, imgs, ids):
		'''
			Expects as input a tensor of B images and a tensor of B image IDs (with B size of the batch, i.e. number of <image, image ID> pairs in the batch)

			The image tensor has a shape [B, C, H, W] where:
			- B = batch size (number of images in the batch)
			- C = number of channels (e.g. 3 channels for RGB)
			- H = height of the image
			- W = width of the image

			The image ID tensor is a tensor of integer digits (or special tokens) with shape [B, N] where:
			- B = batch size (number of images in the batch)
			- M = number of digits in the image ID until now (starts with the start token, might not end with the end token, and might have padding tokens after the end token)
		'''

		# Preprocess input
		imgs = self.img_to_patch(imgs, self.patch_size).to(self.device)		# Convert the input images into patches
		ids = ids.to(self.device)	# Convert the image IDs into a tensor of integers
		B, T, V = imgs.shape	# B is the batch size (number of images in the batch), T is the total number of patches of the image, and V is the size of the patches' vectors (flattened into value of each color channel, per width, per height)
		imgs = self.input_layer(imgs.to(self.device)) # Convert the input images' patches into embeddings, i.e. vectors (one for each patch) of size "embed_dim"

		# Convert the image IDs into embeddings
		M = ids.shape[1]				# The number of digits in the current (possibly incomplete, hence M<N) image ID given as input to the model
		N = self.img_id_max_length + 2		# The maximum number of digits in the image ID (plus the start and end tokens)

		# Convert each digit of the image ID into an embedding (i.e. a vector of size "embed_dim")
		ids = self.id_embedding(ids).to(self.device)	# Shape: [B, M, embed_dim]

		# Concatenate the image embeddings with the image ID embeddings
		# - imgs size: [B, T, embed_dim]
		# - ids size: [B, M, embed_dim]
		x = torch.cat([imgs, ids], dim=1)

		# Convert the input tensor to a float tensor and move it to the model's device
		x = x.to(self.device, dtype=torch.float)

		# Complete the image ID embeddings with padding tokens
		# - If the image ID has less than the maximum number of digits, pad the remaining digits
		# - If the image ID has more digits than the maximum number of digits, truncate it
		padding_sequence = None
		if M < N:
			padding_sequence = torch.full((B, N - M, self.embed_dim), self.img_id_padding_token, dtype=torch.float, device=self.device)
			x = x.to(self.device, dtype=torch.float)
			padding_sequence = padding_sequence.to(self.device, dtype=torch.float)
			x = torch.cat([x, padding_sequence], dim=1)
		if M > N:
			x = x[:, : N]

		# Add positional encoding at the end of each sequence
		# NOTE: new shape of x after positional encodings is [B, T + N, embed_dim]
		if self.learn_positional_encodings:
			x = x + self.pos_embedding[:, : T + N].to(x.device)	# Add positional encoding at the end of the sequence
		else:
			x = self.pos_embedding(x).to(self.device)	# Add positional encoding at the end of the sequence


		# Get a mask for the image ID embeddings
		# - The mask is True for the padding tokens and False for the other tokens
		# - The mask is used to avoid the Transformer to consider the padding tokens in the computation
		# NOTE: the padding mask should be a 2D tensor of shape [B, T + N] (T is the total number of patches in the image and N is the maximum number of digits in the image ID) containing True for the padding tokens and False for the other tokens
		padding_mask = torch.full((B, T + N), False, dtype=torch.bool, device=self.device)
		padding_mask[:, T + M:] = True

		# Get a mask for the attention mechanism (i.e. mask the future tokens) from the masking sequence
		# NOTE: the attention mask should be a 2D tensor of shape [T + N, T + N] (T is the total number of patches in the image and N is the maximum number of digits in the image ID) containing True for the future tokens and False for the current and past tokens
		attention_mask = nn.Transformer.generate_square_subsequent_mask(T + N, device=self.device)	

		# Apply dropout to the input tensor
		if self.dropout.p > 0.0:
			x = self.dropout(x)
		# Compute the Transformer's output given the input tensor x (along with the padding mask and the attention mask)
		x = x.transpose(0, 1)
		transformer_input = (x, padding_mask, attention_mask)	# The transformer (hence the first "attention block" layer of the transformer) expects a tuple of three elements: the input tensor, the padding mask, and the attention mask
		ret_tuple = self.transformer(transformer_input)			# The transformer's output is a tuple of three elements: the output tensor, the padding mask, and the attention mask
		x = ret_tuple[0].transpose(0, 1)	# The output tensor is the first element of the tuple, hence we transpose it to have the shape [B, T + N, embed_dim]

		# Perform classification prediction
		# NOTE: The last element of the output is the "class" token, i.e. in this case the last token of the image ID (the predicted token digit given an image and the start digits of the token ID)
		encoded_digit = x[:, -1, :]		# Shape: [B, embed_dim]
		out = self.mlp_head(encoded_digit).to(self.device) 	# The output is the result of the final MLP head (i.e. the classification layer), hence is a tensor of shape [B, num_classes]

		# Return the logits for the next image ID digit prediction, with N possible classes (10 digits + 2 special tokens, without considering the "Begin of Sequence" token)
		return out

	# Auxiliary function for both the training and valdiation steps (to compute the loss and accuracy)
	def step(self, batch : tuple[torch.Tensor, torch.Tensor], use_autoregression=False):
		'''
		Generate the output image ID using an autoregressive approach (i.e. generate the sequence token by token using the model's own predictions) or using a 
			teacher forcing approach (i.e. use the actual masked target sequence as input to the model) based on wheter the transformer is in evaluation mode or training mode.

		Returns the loss and accuracy of the model for the given batch
		'''
		# Get the input and target sequences from the batch
		input, target = batch	# input is the image tensor of shape [B, C, H, W], target is the image ID tensor of shape [B, N]
								# - B is the batch size (number of images in the batch)
								# - C is the number of channels (e.g. 3 channels for RGB)
								# - H is the height of the image
								# - W is the width of the image
								# - N is the maximum number of digits in the image ID
		input = input.to(self.device)
		target = target.to(self.device)
		B, C, H, W = input.shape
		N = self.img_id_max_length + 2	# The maximum number of digits in the image ID (plus the start and end tokens)
		# Initialize the output tensor (i.e. the final image ID prediction), which should have a shape of [B, N, num_classes], i.e. outputs all the classes/digits for each position/digit in the image ID
		# NOTE: the output will contain the logits for all the possible classes tokens at all the possible digits position, where each digit position only contains the possible digit's logits of the 
		# 		best previous token, or of the ground truth previous token: this means that taking e.g. the second best token of a digit and then appending the best next token won't make much sense,
		# 		since the next token would be based on the best previous token, not the second best previous token...
		output = torch.zeros((B, N-1, self.num_classes), device=self.device)
		# Start with the first token (start token) for all the sequences in the batch (shape: [B, 1])
		generated_target = target[:, 0].unsqueeze(1)	# The target_in is the input sequence for the model, i.e. the sequence of tokens that the model should predict
		# Iterate over the target sequence to generate the output sequence
		for i in range(1, N):
			# Store the next token
			next_token = None
			# Check if the autoregressive approach should be used
			if use_autoregression:
				# Compute the next token's logits using the input and the target sequences, thus relying only on the model's image ID digits predictions ("auto-regressive" approach, "AR")
				classes_predictions_ar = self(input, generated_target)		# Shape: [B, num_classes]
				# Append the last token prediction to the output tensor
				output[:, i - 1] = classes_predictions_ar
				# Use the last generated best token (i.e. token with the highest logit) as the next token of the generated_target sequence
				next_token = torch.argmax(classes_predictions_ar, dim=-1).unsqueeze(1)	# Shape: [B, 1]
				# Append the next token to the generated_target sequence
				generated_target = torch.cat((generated_target, next_token), dim=1)
			else:
				# Get the ground truth target sequence up until the current position "i" (for all batches, shape: [B, i])
				current_target = target[:, :i]	# Shape: [B, i]
				# Compute the next token's logits using the input and the ground truth target sequence ("teacher forcing" approach, "TF")
				classes_predictions_tf = self(input, current_target)	# Shape: [B, num_classes]
				# Append the tokens to the output tensor
				output[:, i - 1] = classes_predictions_tf
		# Get the target output, i.e. the complete image ID (excluding the first token, i.e. the start token)
		target_output_ids = target[:, 1:]	# Shape: [B, N-1]
		# Ensure the target_out tensor is contiguous in memory (to efficiently compute the loss)
		target_output_ids = target_output_ids.contiguous()
		# Compute the loss as the cross-entropy loss between the output and the target_out tensors, i.e. the full predicted image ID and the actual image ID (IDs are encoded, hence are tensors of N digits)
		reshaped_output = output.view(-1, self.num_classes)		# Shape: [B*(N-1), num_classes]
		reshaped_target = target_output_ids.view(-1)			# Shape: [B*(N-1)]
		loss = functional.cross_entropy(reshaped_output, reshaped_target, ignore_index=self.img_id_padding_token)		# Compute the cross-entropy loss
		# Get the best prediction (to compute the accuracy) for the next token of the target sequence (i.e. the generated image ID token/digit)
		predictions = torch.argmax(output, dim=-1)
		# Compute accuracy with masking for padding
		non_padding_mask = (target_output_ids != self.img_id_padding_token)
		num_correct = ((predictions == target_output_ids) & non_padding_mask).sum().item()
		num_total = non_padding_mask.sum().item()
		accuracy_value = num_correct / num_total if num_total > 0 else 0.0
		accuracy = torch.tensor(accuracy_value)
		# Return loss and accuracy (as tensors)
		return loss, accuracy

# Attention Block for the Vision Transformer model
class AttentionBlock(nn.Module):

	def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
		"""
		Custom attention block for the vision transformer.

		Args:
			embed_dim: Dimensionality of input and attention feature vectors
			hidden_dim: Dimensionality of hidden layer in feed-forward network (usually 2-4x larger than embed_dim)
			num_heads: Number of heads to use in the Multi-Head Attention block
			dropout: Amount of dropout to apply in the feed-forward network
		"""
		super().__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.layer_norm_1 = nn.LayerNorm(embed_dim, device=self.device)
		self.layer_norm_1 = self.layer_norm_1.to(self.device)
		self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, device=self.device)
		self.attn = self.attn.to(self.device)
		self.layer_norm_2 = nn.LayerNorm(embed_dim, device=self.device)
		self.layer_norm_2 = self.layer_norm_2.to(self.device)
		self.linear = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim, device=self.device),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, embed_dim, device=self.device),
			nn.Dropout(dropout),
		)
		self.linear = self.linear.to(self.device)

	def forward(self, x):
		# Takes as input a tuple x of three elements:
		# - x[0] is the input tensor (B, T, V) where:
		#	- B is the batch size
		#	- T is the number of tokens in the sequence
		#	- V is the size of the token vectors
		# - x[1] is the padding mask for the input tensor
		# - x[2] is the attention mask for the input tensor
		input = x[0].to(self.device)
		padding_mask = x[1].to(self.device)
		attention_mask = x[2].to(self.device)
		inp_x = self.layer_norm_1(input).to(self.device)
		input = input + self.attn(inp_x, inp_x, inp_x, key_padding_mask=padding_mask, attn_mask=attention_mask)[0].to(self.device)
		input = input + self.linear(self.layer_norm_2(input)).to(self.device)
		# Return the output tensor (a tuple of three elements: the output tensor, the padding mask, and the attention mask)
		return (input, padding_mask, attention_mask)

