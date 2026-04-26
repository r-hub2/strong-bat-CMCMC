# CUDA_HOME environment variable must be set before entry

CFLAGS2 = -I$(CUDA_HOME)/include -DADD_

libcmcmc.a: cmcmc.o inca.o covariance.o generate_mvn.o autotune.o savesamps.o metrop_1gpu_sm30.o
	$(AR) rcs libcmcmc.a cmcmc.o inca.o covariance.o generate_mvn.o autotune.o savesamps.o metrop_1gpu_sm30.o

cmcmc.o: cmcmc.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c cmcmc.c

inca.o: inca.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c inca.c

covariance.o: covariance.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c covariance.c

generate_mvn.o: generate_mvn.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c generate_mvn.c

autotune.o: autotune.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c autotune.c

savesamps.o: savesamps.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c savesamps.c

metrop_1gpu_sm30.o: metrop_1gpu_sm30.c
	$(CC) $(CFLAGS) $(CFLAGS2) -c metrop_1gpu_sm30.c

clean:
	rm -f *.o *.dylib *.so *.a cmcmc
