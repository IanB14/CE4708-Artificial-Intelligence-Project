rem - Setup Script - Lorcan Chinnock 09/10/2017
@echo on
@"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
choco install -y wget git python2 7zip.commandline
C:\Python27\Scripts\easy_install pip
rem Download SuiteSparse source
wget http://faculty.cse.tamu.edu/davis/SuiteSparse/SuiteSparse-4.5.4.tar.gz
7z x SuiteSparse-4.5.4.tar.gz
7z x SuiteSparse-4.5.4.tar
set CVXOPT_SUITESPARSE_SRC_DIR=%CD%\SuiteSparse

rem Download compatible OpenBLAS library
wget https://bitbucket.org/carlkl/mingw-w64-for-python/downloads/OpenBLAS-0.2.17_amd64.7z
mkdir openblas
7z x OpenBLAS-0.2.17_amd64.7z -aoa -oopenblas

rem Install mingwpy Python extension using pip
pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy

rem Clone CVXOPT repository, configure, compile, and install
git clone https://github.com/cvxopt/cvxopt.git
copy openblas\amd64\lib\libopenblaspy.a cvxopt\
cd cvxopt
for /f %%a in ('git describe --abbrev^=0 --tags') do git checkout %%a
set CVXOPT_BLAS_LIB=openblaspy
set CVXOPT_LAPACK_LIB=openblaspy
set CVXOPT_BLAS_LIB_DIR=%CD%
set CVXOPT_BLAS_EXTRA_LINK_ARGS=-lgfortran;-lquadmath
python setup.py build --compiler=mingw32
python setup.py install
python -m unittest discover -s tests

ECHO 'Done'