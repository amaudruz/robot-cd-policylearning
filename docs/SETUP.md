### Environment Setup

1.  Install Python-RVO2 library
	```
	wget https://github.com/sybrenstuvel/Python-RVO2/archive/master.zip && unzip master && rm master.zip
	cd Python-RVO2-master && pip install Cython && pip install -r requirements.txt
	python setup.py build && python setup.py install
	```

2.  Install CrowdNav environment
	```
	pip install -e .
	```

3.  Download data
	```
	cd crowd_nav && mkdir data/demonstration/ -p && cd data/demonstration
	pip install gdown && gdown https://drive.google.com/uc?id=1D2guAxD_EgrKnJFMcLSBkf10SOagz0mr
	```
	(Optional) Generate data
	```
	cd crowd_nav && mkdir data/expert/ -p && cd data/expert
	gdown https://drive.google.com/uc?id=1awXDsRQcmgacj7nUhPzwb5UMNZCJCjvu
	python utils/demonstrate.py --policy="sail" --output_dir="data/output/demonstration" --memory_dir="data/demonstration" --expert_file="data/expert/rl_model.pth"
	python utils/convert.py
	```
