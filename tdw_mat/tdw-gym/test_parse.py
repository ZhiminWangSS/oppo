from lm_agent_cobel import lm_agent_cobel


txt = "<purser>(1000)"


s1,s2,s3 = lm_agent_cobel.parse_entity(txt)

print(s1,s2,s3)