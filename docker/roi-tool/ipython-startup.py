

import hcpannot
from hcpannot import ROITool

# A hack to pull the github username out of the git directory
# and put it where ROITool can find it.
import os
git_username = os.popen("grep 'url = https://github.com/' ~/.extgit/config").read()
git_username = git_username.split('github.com/')[-1].split('/hcp-annot-vc')[0]
if git_username is None or git_username == '':
    git_username = os.popen("grep 'url = git@github.com:' ~/.extgit/config").read()
    git_username = git_username.split('github.com:')[-1].split('/hcp-annot-vc')[0]
os.environ['GIT_USERNAME'] = git_username
# Make sure there is a directory for this user to save into.
if not os.path.isdir(f'./save/{git_username}'):
    os.makedirs(f'./save/{git_username}', mode=0o755)





    
