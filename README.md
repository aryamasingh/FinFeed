# FinFeed
An AI assisted resource (a conversational bot) that efficiently collates finance and economy related current news within a specified time period (eg. 1 week) from a YouTube news channel (eg. Bloomberg) where people can ask about finance related current news.
# Authors: 
Aryama Singh (as3844@cornell.edu)
# Moelling Approach
![Example Image](dataflow2.png)

## FinFeed conda environment

If you want the most streamlined expereince possible this semester, you should set up a finfeed conda environment and run all of the notebooks with this environment.

Check to make sure you have conda by running the following in your command line interface:

    conda --version

If you don't have conda, google how to install it!

Once you have conda run:

    conda env create --name finfeed_env --file=finfeed_env.yml

Press [y] to all of the prompts.  You will be downloading a lot of packages.

Once this is done:

    conda activate finfeed_env

To check everything is there:

    conda list

Should show all of the packages!