# TODO (ordered by priority)
- cmf_tensor_free unconditionaly free t->data but a tensor created via cfm_tensor_from does not owns t->data
- implement cfm_tensor_cat, cfm_tensor_expand, and the python tensor[-1] equivalent
- requires_grad currently does not implies/do nothing, need to work on the grad stuff.
