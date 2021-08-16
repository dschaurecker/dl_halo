import torch


def exp(x, undo=False):
    if not undo:
        torch.exp(x, out=x)
    else:
        torch.log(x, out=x)

def log(x, eps=1e-8, undo=False):
    if not undo:
        torch.log(x + eps, out=x)
    else:
        torch.exp(x, out=x)

def expm1(x, undo=False):
    if not undo:
        torch.expm1(x, out=x)
    else:
        torch.log1p(x, out=x)

def log1p(x, eps=1e-7, undo=False):
    if not undo:
        torch.log1p(x + eps, out=x)
    else:
        torch.expm1(x, out=x)

def myexp(x, undo=False):
    if not undo:
        torch.exp(x, out=x)
    else:
        torch.log(x, out=x)

def mylog(x, eps=1e-3, undo=False):
    if not undo:
        torch.log(x + eps, out=x)
    else:
        torch.exp(x, out=x) - eps

def myexpm1(x, undo=False):
    if not undo:
        torch.expm1(x, out=x)
    else:
        torch.log1p(x, out=x)

def mylog1p(x, eps=0, undo=False):
    if not undo:
        torch.log1p(x + eps, out=x)
    else:
        torch.expm1(x, out=x) - eps

def myloglog1p(x, eps=1e-7, undo=False):
    if not undo:
        mylog1p(x)
        mylog(x)
    else:
        mylog(x, undo=True)
        mylog1p(x, undo=True)
        
def myloglog(x, eps=1e-3, undo=False):
    if not undo:
        torch.log(x+eps, out=x)
        torch.log((-1)*x, out=x)
    else:
        torch.exp(x, out=x)
        torch.exp((-1)*x, out=x) - eps
        
def logmylog(x, undo=False):
    if not undo:
        mylog(x)
        torch.log(-1*x, out=x)
    else:
        torch.exp(x, out=x)
        mylog(-1*x, undo=True)