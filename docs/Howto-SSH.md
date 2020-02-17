<div style="text-align: left">
    <img src="../fidle/img/00-Fidle-header-01.svg" style="width:800px"/>
</div>


# How to  - SSH for GRICAD
How to configure your SSH environment for friendly access to GRICAD in 6 steps.

## Step 1 - Generate yours ssh keys
If you don't already have SSH keys, create one with a nice password :

```
# ssh-keygen
```

## Step 2 - ssh-agent configuration
To avoid having to enter your ssh password all the time, we will use the ssh-agent service.  
Try :
```
# ssh-add
```
If an error occurs, add this lines in your .bash_profile and restart your session :
```
pgrep ssh-agent >/dev/null || ssh-agent -s > ~/.ssh_agent
. ~/.ssh_agent
ssh-add -l >/dev/null || ssh-add
```

## Step 3 - Allow your access to the bastions
The password requested is your PERSEUS password.
\<login\> is your PERSEUS  login
```
# ssh-copy-id <login>@rotule.imag.fr
```
and
```
# ssh-copy-id <login>@trinity.ujf-grenoble.fr
```
To verify, these two commands must now work :
```
# ssh <login>@rotule.imag.fr hostname
# ssh <login>@trinity.ujf-grenoble.fr hostname
```

## Step 4 - Configuring access through bastions
Modify (or create if it doesn't exist) your **.ssh/config** file, with :
```
ForwardAgent yes

Host *.ciment
  User <login>
  ProxyCommand ssh -q <login>@access-rr-ciment.imag.fr "nc -w 60 `basename %h .ciment` %p"
  LocalForward 8888 f-dahu:<your uid>
  LocalForward 6006 f-dahu:<your uid + 10000>
```
Where :
  - \<login\> is your PERSEUS login
  - \<your uid\> is your uid on rotule or trinity
  - \<your uid + 10000\> is your uid + 10000  (if uid=6500, that makes 16500)

  To get your uid, try : `# ssh <your login>@rotule.imag.fr id -u`

## Step 5 - Drink a coffee 
You've earned it, but courage, it's almost over !

## Step 6 - Allow your access to the frontal
As before, the password requested is your PERSEUS password.
```
# ssh-copy-id f-dahu.ciment
```
**Fine !** Normalement, on peut maintenant accéder directement à la frontale - de manière sécurisée et simple, - sans ressaisir 18 fois son mot de passe :-)  
**To check :**
```
# ssh f-dahu.ciment hostname
f-dahu
```
If that doesn't work, drink another coffee and check your steps...
---
<div style="text-align: left">
    <img src="../fidle/img/00-Fidle-logo-01.svg" style="width:80px"/>
</div>