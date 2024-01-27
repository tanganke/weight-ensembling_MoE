R"""
Algorithm 1 Pseudocode of MoCo in a PyTorch-like style. 

```python
# f_q, f_k: encoder networks for query and key 
# queue: dictionary as a queue of K keys (CxK) 
# m: momentum 
# t: temperature 
f_k.params = f_q.params # initialize
for x in loader: # load a minibatch x with N samples 
    x_q = aug(x) # a randomly augmented version 
    x_k = aug(x) # another randomly augmented version 
    
    q = f_q.forward(x_q) # queries: NxC 
    k = f_k.forward(x_k) # keys: NxC 
    k = k.detach() # no gradient to keys 
    
    # positive logits: Nx1 
    l_pos = bmm(q.view(N,1,C), k.view(N,C,1)) 
    
    # negative logits: NxK 
    l_neg = mm(q.view(N,C), queue.view(C,K)) 
    
    # logits: Nx(1+K) 
    logits = cat([l_pos, l_neg], dim=1) 
    
    # contrastive loss, Eqn.(1) 
    labels = zeros(N) # positives are the 0-th 
    loss = CrossEntropyLoss(logits/t, labels) 
    
    # SGD update: query network 
    loss.backward() 
    update(f_q.params) 
    
    # momentum update: key network 
    f_k.params = m*f_k.params+(1-m)*f_q.params 
    
    # update dictionary 
    enqueue(queue, k) # enqueue the current minibatch
    dequeue(queue) # dequeue the earliest minibatch 
```

bmm: batch matrix multiplication; mm: matrix multiplication; cat: concatenation.
"""
