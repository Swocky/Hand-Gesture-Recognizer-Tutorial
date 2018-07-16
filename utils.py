def remove_module_from_checkpoint_state_dict(state_dict):
    """
    Removes the prefix `module` from weight names that gets added by
    torch.nn.DataParallel()
    """
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict