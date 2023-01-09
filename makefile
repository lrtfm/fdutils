
all: modules

modules:
	@echo "    Building extension modules"
	@python setup.py build_ext --inplace > build.log 2>&1 || cat build.log

develop: clean
	@echo "    Develop the extension"
	@python -m pip install -r requirements.txt > install_requirements.log 2>&1 || cat install_requirements.log
	@# @python -m pip install -vvv -e . > develop.log 2>&1 || cat develop.log
	@python -m pip install --no-build-isolation --no-binary mpi4py,randomgen,islpy,numpy --no-deps -vvv -e . > develop.log 2>&1 || cat develop.log

clean:
	@echo "    Cleaning extension modules"
	@python setup.py clean > /dev/null 2>&1
	@echo "    RM peval/evalpatch.*.so"
	-@rm -f fdutils/evalpatch.*.so > /dev/null 2>&1
	@echo "    RM peval/evalpatch.c"
	-@rm -f fdutils/evalpatch.c > /dev/null 2>&1
	@echo "    RM peval.egg-info"
	-@rm -rf fdutils.egg-info > /dev/null 2>&1

