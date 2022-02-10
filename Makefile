
clean-pyc:
	find ngv_ctools -type f -name "*.py[co]" -o -name __pychache__ -exec rm -rf {} +

clean-cpp:
	find ngv_ctools -type f -name '*.o' -delete -o -name '*.so' -delete

clean-build:
	rm -rf build dist *.egg-info


.PHONY: clean
clean: clean-build clean-cpp clean-pyc

.PHONY: install
install: clean
	pip3 install --upgrade setuptools pip
	pip3 install -e .
