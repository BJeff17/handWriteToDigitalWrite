
def train(model, data_loader,epochs, loss_fn, optimizer, logger_fn):
    for _ in range(epochs):
        lossG = 0
        t = 0
        for i, data_batch in enumerate(data_loader):
            x_batch, y_batch = data_batch
            out = model(x_batch)
            loss = loss_fn(out, y_batch)
            lossG += loss.item()
            optimizer.zero_grad()
            loss.backward()
            t += data_batch.size()[0]
            optimizer.step()
        lossG /= t
        logger_fn(_, epochs, lossG)


def test(model, data_loader):

    precision = 0

    for i, data_batch in enumerate(data_loader):
        x_batch, y_batch = data_batch
        out = model(x_batch)
        precision = (out == y_batch).sum()/y_batch.size()[0]
        
    return precision
