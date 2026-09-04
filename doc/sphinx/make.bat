@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build
REM The final built HTML lands in doc/html (not doc/sphinx/_build/html)
REM -- see the Makefile's own identical comment.
set HTMLDIR=..\html

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://www.sphinx-doc.org/
	exit /b 1
)

if "%1" == "html" goto html
if "%1" == "clean" goto clean

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:html
%SPHINXBUILD% -b html -d "%BUILDDIR%\doctrees" %SOURCEDIR% %HTMLDIR% %SPHINXOPTS% %O%
goto end

:clean
%SPHINXBUILD% -M clean %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
if exist %HTMLDIR% rmdir /s /q %HTMLDIR%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
