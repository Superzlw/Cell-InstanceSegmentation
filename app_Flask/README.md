# Indruction
This folder contains the relevant files about the model deployment on heroku based on Flask.
## *Flask*


## *gunicorn* (not suitalbe for windows) but *waitress*
Tool that deploys an actual web server that is suitable for production
> **Note**: *fcntl* in *gunicorn* is not supported by windows. As a result, gunicorn cannot be used in windows.

As an alternative to *gunicorn*, you can use *[waitress](https://docs.pylonsproject.org/projects/waitress/en/latest/)*. 
Another reference for that deserves reading is https://stackoverflow.com/questions/11087682/does-gunicorn-run-on-windows.

## *heroku*
Build heroku app based on gunicorn (which is not supported on windows), or on waitress (supported by windows)

> About Install and login
> Reference link: https://devcenter.heroku.com/articles/heroku-cli
> - If your OS is linux, you can directly use `pip install heroku3` in your venv and then use CLI command to perform login.
> - If your OS is windows, you need to download and install the exe provided in the Heroku website and then can use CLI 
    command in your powershell. Note: the built-in terminal supported by PyCharm cannot recognize the *heroku* command, 
    instead you can use the built-in powershell supported by windows or by anaconda. And please don't forget to activate
    your virtual environment.
>
> For login, you can command `heroku login` or  `heroku login -i` if you prefer login within the terminal. 

> **Note**: Since the space offered by heroku is limited,  so cpu-only version of pytorch should be adopted. Accordingly,
> the inference mode of mmdetection in terms of model should be also be set in cpu other than gpu.
> Thus, after when you export the current in-use python packages into *cellis_requirements.txt* for example,
> you can replace `torch==1.6.0 torchvision==0.7.0` with `torch==1.6.0+cpu torchvision==0.7.0+cpu` in the 
> *cellis_requirements.txt* and meanwhile locate `-f https://download.pytorch.org/whl/cpu/torch_stable.html` atop in 
> the same txt file.

As lang as the work above is done, you can add this as a remote repo for heroku by commanding `heroku git:remote -a <app name>`,
where you can simply name the app name as your project name or something else.

After that, you can then commit and push everything to your github repo. At last, you also need to push them again to heroku 
by commanding `git push heroku master`.