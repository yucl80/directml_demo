from loguru import logger
import time

# 设置API密钥
api_key = "your-api-key"
base_url = "http://127.0.0.1:8000/v1/"

# 设置日志记录
logger.add("code_review.log", rotation="500 MB")

from openai import OpenAI

client = OpenAI(base_url=base_url, api_key=api_key)

def send_llm_request(messages):
    try:
        messages_ = [
             {
                    "role": "system",
                    "content": "作为代码审查助手。我将提供给你变更前后的代码。请你简要总结差异内容，帮助审查者更快更方便地理解文件中的变化。总结必须完全客观，不含意见或建议。",
                },
        ]
        messages_.append(messages)        
        response = client.chat.completions.create(
            # model="bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            model="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
            n=1,
            stop=None,           
        )
        review = response.choices[0]
        return review
    except Exception as e:
        logger.error(f"创建请求时异常: {e}")
        return "请求过程中出现错误。"



def create_change_request(lang, old_code, new_code):
    prompt = f"""
    代码如下：
        旧代码:
        ```{lang}
        {old_code}
        ```\n
        新代码:
        ```{lang}
        {new_code}
        ```      
        """
    return  {"role": "user", "content": prompt}


def create_summary_request(change_list):
    prompt = f"""请务必使用中文回复，不要超过20个字。
        函数及函数的变更描述如下：      
        ```
        {change_list}
        ```\n      
        """
    messages=[
                {
                    "role": "system",
                    "content": "作为代码审查助手。我将提供给你函数的变更描述以。请你简要总结变更描述内容，帮助审查者更快更方便地理解文件中的变化。总结必须完全客观，不含意见或建议。",
                },
                {"role": "user", "content": prompt},
            ]
    return send_llm_request(messages)


def display_review_results(review):
    print("==== 代码审查结果 ====")
    print(review)
    print("========================")


if __name__ == "__main__":
    old_code = """
    public class OrderService{
   private double getOrderAmount(Order order){
        return calcAmout(order);
    }
    }
    """

    new_code = """
     public class OrderService{
   private double getOrderAmount(Order order){
        double amout = calcAmout(order);
        return amout * getOrderDiscounts(amout);
    } 

   
    }
    """   
       
    
    old_code2 = """
    public class OrderDao{
    private double queryOrder(Order order){
        return jdbcTemplate.query(order);
    }
    }
    """

    new_code2 = """
    public class OrderDao{
    private double queryOrder(Order order){
        List<Order> orderList = jdbcTemplate.query(order);
        orderList = orderList.stream().filter(o -> o.getOrderStatus() == OrderStatus.Paid ).collect(Collectors.toList());
        return order;
    }    
    }
    """
    
    lang = "java"
    begin_time = time.time()
    request1 = create_change_request(lang, old_code, new_code)  
    request2 = create_change_request(lang, old_code2, new_code2)  
  
    review = send_llm_request([request1,request2])
    
    t1 =   time.time() - begin_time
    display_review_results(review)     
   
    
    print("used_time:",t1) 
   
    change_list = f"""{review.message.content} """
    # 创建单个API的调用链变更的汇总
    s = create_summary_request(change_list)
    
    print(s)
    
    # 创建全部的变更汇总
    
