import random 

outcomes = [0, 5, 10, 20]
p_r = [0.35, 0.30, 0.25, 0.10]
p_s = [0.30, 0.35, 0.30, 0.05]
p_s2 = [0.25, 0.25, 0.25, 0.25] 
p_s3 = [0.001, 0.499, 0.499, 0.001]
nsamples = 10000

#Exercise 3.2
def makesamples(dist, n):
    return [random.choices(outcomes, dist)[0] for _ in range(n)]

# Exercise 3.3

R_s = makesamples(p_s, nsamples)
E_hat = sum(R_s)/nsamples
print(E_hat)

#Exersice 3.4
def compute_p(outcomes, dist1, dist2):
    outcomes_p = {}
    for o, r, s in zip(outcomes, dist1, dist2):
        prs = r/s
        outcomes_p[o] = prs
    return outcomes_p
p = compute_p(outcomes, p_r, p_s)

print(p)

#Exersize 3.5 
def e_hat_r(R_s, p):
    return sum(x * p[x] for x in R_s) / len(R_s)

e_hatr = e_hat_r(R_s, p)
print(e_hatr)

#Exersice 3.6
p2 = compute_p(outcomes, p_r, p_s2)
R_s2 = makesamples(p_s2, nsamples)
e_hat_s2 = sum(R_s2)/nsamples
e_hatr_s2 = e_hat_r(R_s2, p2)
print("p2: ", p2, ", E-hat[S']: ", e_hat_s2, ", E-hat R wrt s': ", e_hatr_s2)

#Exersice 3.7
p3 = compute_p(outcomes, p_r, p_s3)
R_s3 = makesamples(p_s3, nsamples)
e_hat_s3 = sum(R_s3)/nsamples
e_hatr_s3 = e_hat_r(R_s3, p3)
print("p3: ", p3, ", E-hat[S'']: ", e_hat_s3, ", E-hat R wrt s': ", e_hatr_s3)

       