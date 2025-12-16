import os
import kagglehub


def download_dataset_to(dest_dir=None):
	"""Download the dataset into `dest_dir`. If None, uses current working directory."""
	if dest_dir is None:
		dest_dir = os.getcwd()

	# Try calling dataset_download with a path argument if supported
	try:
		path = kagglehub.dataset_download("soumikrakshit/lol-dataset", path=dest_dir)
		return path
	except TypeError:
		# If the function doesn't accept a path kwarg, change working dir temporarily
		prev = os.getcwd()
		os.makedirs(dest_dir, exist_ok=True)
		os.chdir(dest_dir)
		try:
			path = kagglehub.dataset_download("soumikrakshit/lol-dataset")
		finally:
			os.chdir(prev)
		return path


if __name__ == "__main__":
	# download into the directory where this script is run
	path = download_dataset_to()
	print("Path to dataset files:", path)