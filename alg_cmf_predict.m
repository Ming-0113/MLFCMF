function [A,B]=alg_cmf_predict(Y,A,B,Sd,St,lambda_l,lambda_d,lambda_t,num_iter,W)

    K = size(A,2);    
                       alpha = 0.5;
        
        Sd0 = Sd - diag(diag(Sd));
        Sd1 = (Sd0 + Sd0') / 2; 
        DM = diag(sum(Sd1));
        Sd2 = diag(1 ./ diag(sqrtm(DM)));      
        Sd=SKF({Sd0,Sd2},54,10,0.1);	
        Sd = alpha*Sd + (1-alpha)*getGipKernel(Y);    
    
        St0 = St - diag(diag(St));
        St1 = (St0 + St0') / 2; 
        DD = diag(sum(St1));
        St2 = diag(1 ./ diag(sqrtm(DD)));
        St=SKF({St0,St2},26,10,0.1);  
        St = alpha*St + (1-alpha)*getGipKernel(Y');     
     
	lambda_d_Sd = lambda_d*Sd;
    lambda_t_St = lambda_t*St;
    lambda_l_eye_K = lambda_l*eye(K);


 

    if nargin < 10
        AtA = A'*A;
        BtB = B'*B;
        for z=1:num_iter
            A = (Y*B + lambda_d_Sd*A)  / (BtB + lambda_l_eye_K + lambda_d*(AtA));
           
            AtA = A'*A;
            B = (Y'*A + lambda_t_St*B) / (AtA + lambda_l_eye_K + lambda_t*(BtB));
         
            BtB = B'*B;

        obj(z)=(norm(Y-A*B','fro'))^2;
        end
        
    else
        H = W .* Y;
        for z=1:num_iter
            A_old = A;
            HB_plus_lambda_d_Sd_A_old = H*B + lambda_d_Sd*A_old;
            lambda_l_eye_k_plus_lambda_d_A_oldt_A_old = lambda_l_eye_K + lambda_d*(A_old'*A_old);
            for a=1:size(A,1)
                A(a,:) = HB_plus_lambda_d_Sd_A_old(a,:) / (B'*diag(W(a,:))*B + lambda_l_eye_k_plus_lambda_d_A_oldt_A_old);
            end
            B_old = B;
            HtA_plus_lambda_t_St_B_old = H'*A + lambda_t_St*B_old;
            lambda_l_eye_k_plus_lambda_t_B_oldt_B_old = lambda_l_eye_K + lambda_t*(B_old'*B_old);
            for b=1:size(B,1)
                B(b,:) = HtA_plus_lambda_t_St_B_old(b,:) / (A'*diag(W(:,b))*A + lambda_l_eye_k_plus_lambda_t_B_oldt_B_old);
            end


        end
    end
   
end