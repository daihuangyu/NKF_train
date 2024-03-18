import torch
import importlib
import os



def savecheckpoint(model, epoch,  step, loss, optimizer, model_dir):
    checkPath = os.path.join(model_dir, 'checkpoint_'+str(epoch+1) + '_' + str(step+1)+'.pth')
    torch.save({'epoch': epoch,
                'step': step,
                'loss': loss,
                'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
                }, checkPath)
    ckpt_name = 'checkpoint_'+str(epoch+1) + '_' + str(step+1)+'.pth'
    f = open(os.path.join(model_dir, 'checkpoint.txt'), 'w')
    f.write(ckpt_name)
    f.close()
    print('Saving Checkpoint')




def reload_model_val(model, optimizer, model_dir):
    if os.path.isfile(os.path.join(model_dir, 'checkpoint.txt')):
        f = open(os.path.join(model_dir, 'checkpoint.txt'))
        ckpt_name = f.read()
        model_dir = os.path.join(model_dir, str(ckpt_name))
        print(model_dir)
        if os.path.isfile(model_dir):
            print("=> loading checkpoint '{}'".format(model_dir))
            checkpoint = torch.load(model_dir)
            #model.load_state_dict(checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            loss = checkpoint['loss']
            return epoch, step, 100000
    else:
        print('Run new model')
        epoch = -1
        step = 0
        loss = 100000
        return epoch, step, loss

def reload_model_evl(model, model_dir):
    if os.path.isfile(os.path.join(model_dir, 'checkpoint.txt')):
        f = open(os.path.join(model_dir, 'checkpoint.txt'))
        ckpt_name = f.read()
        model_dir = os.path.join(model_dir, str(ckpt_name))
        print(model_dir)
        if os.path.isfile(model_dir):
            print("=> loading checkpoint '{}'".format(model_dir))
            checkpoint=torch.load(model_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def initialize_config(module_cfg, pass_args=True):
    """
    According to config items, load specific module dynamically with params.
    eg，config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])








