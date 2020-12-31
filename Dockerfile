FROM rocker/verse:latest

# Installs additional packages

RUN r -e "install.packages(c(\"estimatr\", \
                             \"minpack.lm\"))"
